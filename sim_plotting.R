rm(list = ls())
library(tidyverse)
library(lme4)

outcomes <- c("enjoy_num","persuasive_num","articulated","informative","aggressive","persuasive_me")
outcomes_expanded <- c( paste0(outcomes, 'Conf_conf'), paste0(outcomes, 'Conf_original'))



# initialize dataframe for storing results
results <- data.frame()
ti_estimator_ps <- data.frame()
cbw_estimator_df <- data.frame()

# for each file in ./qlevel_results_confounded
files <- list.files(path = "./qlevel_results_confounded/", pattern = "*.csv", full.names = TRUE)
counter <- 0
for (file in files) {
    print(counter/length(files))
    data <- read.csv(file)
    # mutate new column for weight, which is the number of rows per qid, ih_final
    data <- data %>%
        group_by(qid) %>%
        mutate(weight = n()) %>%
        ungroup()
    
    # check the seed in the file name. strip away the "seed" and ".csv" from the file name and convert to numeric
    seed <- as.numeric(gsub("seed|\\.csv", "", basename(file)))
    # for each outcome in outcomes_expanded
    for (outcome in outcomes_expanded) {
        # run a regression with ih_final as the treatment, qid as a fixed effect, ResponseId as a random effect, and weight as the weight and outcome as the outcome
        # store the coefficient for ih_final
        formula <- as.formula(paste(outcome, "~ ih_final + (1 | ResponseId) + qid"))
        model <- lmer(formula, data = data, weights = weight)
        coef_ih_final <- fixef(model)["ih_finaly"]
        
        # read in results for each causal-text baseline: ./causal-text/src/sim_data/seed{seed}/outcome_{outcome}_tx_effects_point_estimate.csv
            # compute average of each column across all rows
        causal_text_file <- paste0("./causal-text/src/sim_data/seed", seed, "/outcome_", outcome, "_tx_effects_point_estimate.csv")
        causal_text_results <- read.csv(causal_text_file) %>%
            summarise(across(everything(), function(x) mean(x, na.rm = TRUE)))

        # read in results for each TI_estimator baseline: ./TI_estimator/src/sim_data/seed{seed}/outcome_{outcome}_tx_effects_point_estimate.csv
        cbw_text_file <- paste0("./causal-text/src/sim_data/seed", seed, "/outcome_", outcome, "_cbw_components.csv")
        # join dataframe for TI estimator baselines with causal-text baselines
        # mutate a new column for the regression coefficient
        # bind the results to the dataframe for storing results
        cbw_text_components <- read_csv(cbw_text_file, show_col_types = FALSE) %>%
            mutate(seed = seed, outcome = outcome) %>%
            select(Q1, Q0, seed, outcome)
        cbw_estimator_df <- bind_rows(cbw_estimator_df, cbw_text_components)
        
        ti_estimator_file <- paste0("./TI_estimator/src/sim_data/seed", seed, "/outcome_", outcome, "_tx_effects_point_estimate.csv")
        ti_estimator_results <- tryCatch({
            read.csv(ti_estimator_file)
        }, error = function(e) {
            data.frame(ti_aipw_trimmed = NA, ti_aipw_winsorized = NA, ti_or = NA)
        })
        
        # combine the results for the TI estimator and causal-text baselines
        combined_results <- bind_cols(ti_estimator_results, causal_text_results) %>%
            mutate(true_ate = coef_ih_final) %>%
            select(-seed) %>%
            mutate(seed = seed) %>%
            mutate(outcome = outcome) %>%
            separate(outcome, sep = 'Conf_', into = c('outcome', 'conf'))
        
        # get propensity scores for estimators
        results <- bind_rows(results, combined_results)
        ti_aipw_file <- paste0("./TI_estimator/src/sim_data/seed", seed, "/outcome_", outcome, "_aipw_components.csv")
        ti_estimator_ps <- bind_rows(ti_estimator_ps, read_csv(ti_aipw_file, show_col_types = FALSE) %>% mutate(seed = seed, outcome = outcome))

        ipw_file <- paste0("./causal-text/src/sim_data/seed", seed, "/outcome_", outcome, "_ps_bow.csv")
        ipw_results <- read_csv(ipw_file, show_col_types = FALSE) %>%
            mutate(seed = seed, outcome = outcome)
    }
    # update counter
    counter <- counter + 1
}


# create a new dataframe with the new names for the estimators
new_names <- tribble(
        ~name, ~new, ~est_order,
        'unadj_T_proxy' , 'Difference in means', 1,
        'ate_T_proxy' , 'Topic adjustment', 2,
        'ate_bow', 'BoW OR', 3,
        'ate_aipw', 'BoW AIPW', 4,
        'ate_ipw', 'BoW IPW', 3.5,
        'ti_aipw_winsorized', ' TI Winsorized', 5,
        'ti_aipw_trimmed', 'TI Trimmed', 5.5,
        'ate_cb_T_proxy' , 'TextCause', 6,
        'true_ate', 'True ATE', 7
) %>% mutate(new = factor(new))


results_long <- results %>%
    group_by(outcome, conf) %>%
    mutate(ptile025 = quantile(true_ate, 0.025),
           ptile975 = quantile(true_ate, 0.975)
    ) %>%
    pivot_longer(cols = -c(outcome, conf, seed, iter, ptile025, ptile975)) %>%
    inner_join(new_names, by = c('name' = 'name')) %>%
    mutate(conf = case_match(
        conf, 
        'conf' ~ 'Amplified Confounding', 
        'original' ~ ' Baseline Confounding')
        ) %>%
    mutate(outcome = case_match(outcome,
                                    'persuasive_me' ~ 'persuades\nme',
                                    'persuasive_num' ~ 'persuades\nothers',
                                    'enjoy_num' ~ 'enjoyable',
                                    .default = outcome
        ))

png('./figures/sim_data_results.png', res = 300, height = 12, width = 20, units = 'in')
ggplot(results_long) +

    geom_rect(aes(xmin = ptile025, xmax = ptile975), 
                  ymin = 0,
                  ymax = nrow(new_names) + 1,
                  fill = 'grey', alpha = 0.05) +
    geom_boxplot(aes(y = reorder(new, est_order), x = value, color = new), size = 1.25) +
    facet_grid(conf~outcome, scales = 'free_y') +
    scale_y_discrete(aes(limits = est_order)) +
    geom_vline(aes(xintercept = 0), linetype = 'dashed') +
    scale_x_continuous(breaks = c(0, 0.5, 1)) +
    labs(x = 'Estimated ATE', y = NULL) +
    # facet_wrap(outcome) +
    theme_bw() +
    # theme(axis.text.x = element_text(size = 18))
    theme(text = element_text(size = 36),
            axis.title.y = element_blank(),
            legend.position = 'none')
dev.off()

# figure out how often the propensity score is extreme
ti_estimator_ps %>% 
mutate(outcome = case_match(outcome,
                                    'persuasive_me' ~ 'persuades me',
                                    'persuasive_num' ~ 'persuades others',
                                    'enjoy_num' ~ 'enjoyable',
                                    .default = outcome
        )) %>%
drop_na() %>%
    separate(outcome, sep = 'Conf_', into = c('outcome', 'conf')) %>%
    filter(conf == 'conf') %>%
    filter(seed == 2024) %>%
    mutate(conf = case_match(conf, 'conf' ~ 'Confounded outcome', 'gc' ~ 'GC', 'no' ~ 'No Confounding', 'original' ~ ' Original outcome')) %>%
    ggplot(.) +
    geom_histogram(aes(x = g), bins = 20) +
    facet_wrap(~outcome, scales = 'free_y')

# figure out how often the propensity score is extreme for a single run with seed==2024
ps_violation_table <- ti_estimator_ps %>% 
separate(outcome, sep = 'Conf_', into = c('outcome', 'conf')) %>%
mutate(outcome = case_match(outcome,
                                    'persuasive_me' ~ 'persuades me',
                                    'persuasive_num' ~ 'persuades others',
                                    'enjoy_num' ~ 'enjoyable',
                                    .default = outcome
        )) %>%
drop_na() %>%
    filter(conf == 'conf') %>%
    group_by(seed, outcome) %>%
    summarize(`Est Prop \\leq 0.1` = sum(g <= 0.1)/n(),
    `0.1 < Est Prop < 0.9` = sum((g> 0.1) * (g < 0.9))/n(),
    `0.9 \\leq Est Prop` = sum(g >= 0.9)/n()
    ) %>%
    filter(seed == 2024) %>%
    ungroup() %>%
    select(-seed) %>%
    mutate_if(is.numeric, function(x) round(x, 2))
ps_violation_table_ipw <- lapply(
    FUN = function(outcome)
    paste0("./causal-text/src/sim_data/seed2024/outcome_", outcome, "_ps_bow.csv") %>% 
        read_csv(., show_col_types = FALSE) %>%
        mutate(seed = 2024, outcome = outcome),
    X = outcomes_expanded
) %>% 
bind_rows() %>% 
mutate(g = ps) %>% 
separate(outcome, sep = 'Conf_', into = c('outcome', 'conf')) %>%
mutate(outcome = case_match(outcome,
                                    'persuasive_me' ~ 'persuades me',
                                    'persuasive_num' ~ 'persuades others',
                                    'enjoy_num' ~ 'enjoyable',
                                    .default = outcome
        )) %>%
drop_na() %>%
    filter(conf == 'conf') %>%
    group_by(seed, outcome) %>%
    # summarize(`Est Prop ≤ 0.05` = sum(g <= 0.05)/n(),
    # `0.05 < Est Prop < 0.95` = sum((g> 0.05) * (g < 0.95))/n(),
    # `0.95 ≤ Est Prop` = sum(g >= 0.95)/n()
    # ) %>%
    summarize(`Est Prop \\leq 0.1` = sum(g <= 0.1)/n(),
    `0.1 < Est Prop < 0.9` = sum((g> 0.1) * (g < 0.9))/n(),
    `0.9 \\leq Est Prop` = sum(g >= 0.9)/n()
    ) %>%
    filter(seed == 2024) %>%
    ungroup() %>%
    select(-seed) %>%
    mutate_if(is.numeric, function(x) round(x, 2))


library(stargazer)

# Print the LaTeX table to a file
stargazer(ps_violation_table, 
          title = "The proportion of observations for a single run of each outcome with extreme propensity scores.",
          label = "tab:extreme_ps_estimates",
          table.placement = "t",
          header = TRUE,
        #   summary.stat = c("mean", "sd", "min", "max", "median"),
          type = "latex",
          out = "./prop_score_violations.tex",
          summary=FALSE,
          column.labels = colnames(ps_violation_table)
          )


# Print the LaTeX table to a file
stargazer(ps_violation_table_ipw, 
          title = "The proportion of observations for a single run of each outcome with extreme propensity scores estimated using bag of words text representation.",
          label = "tab:extreme_ps_estimates",
          table.placement = "t",
          header = TRUE,
        #   summary.stat = c("mean", "sd", "min", "max", "median"),
          type = "latex",
          out = "./prop_score_violations_bow.tex",
          summary=FALSE,
          column.labels = colnames(ps_violation_table)
          )

# save ITE results for TextCause to ensure that ITE is not degenerate and there is some variation
png('./figures/cbw_ite.png', res = 300, height = 10, width = 16, units = 'in')
cbw_estimator_df %>%
mutate(ite = Q1 - Q0) %>%
separate(outcome, c('outcome', 'conf'), 'Conf_') %>%
mutate(conf = case_match(conf, 'conf' ~ 'Confounded outcome', 'gc' ~ 'GC', 'no' ~ 'No Confounding', 'original' ~ ' Original outcome')) %>%
mutate(outcome = case_match(outcome,
                                'persuasive_me' ~ 'persuades me',
                                'persuasive_num' ~ 'persuades others',
                                'enjoy_num' ~ 'enjoyable',
                                .default = outcome
    )) %>%
ggplot(.) +
geom_histogram(aes(x = ite)) +
geom_vline(xintercept = 0, linetype = 'dashed') +
facet_grid(conf~outcome) +
labs(x = 'Estimated Individual Treatment Effect', y = NULL) +
scale_x_continuous(breaks = c(-0.05, 0, 0.05)) +
theme_bw() +
theme(text = element_text(size = 24),
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        panel.spacing = unit(2, "lines"),
        plot.margin = margin(1,1,1.5,1.2, "cm"),
        legend.position = 'none')
dev.off()