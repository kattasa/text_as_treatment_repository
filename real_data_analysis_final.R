rm(list = ls())

library(tidyverse)
library(stargazer)

directory <- './'

# get response data
outcomes <- c("enjoy_num","persuasive_num","articulated","informative","aggressive","persuasive_me") %>% sort()

new_qlevel_results_weights_full <- read_csv(paste0(directory, 'qlevel_results.csv')) %>%
        group_by(qid, ih_final) %>%
        mutate(weight = n()) %>%
        ungroup() %>% 
        mutate(row_id = row.names(.)) 
new_qlevel_results_weights <- new_qlevel_results_weights_full %>%
        select(all_of(c('row_id', 'w3_qid', 'qid', 'ih_final', 'weight', 'review_topic', 'ResponseId', outcomes))) %>%
        drop_na() 

# get rating data
rating_data_full <- read_csv(paste0(directory, 'rating_data.csv'))
rating_data_nona  <- rating_data_full %>%
    select(statement_id, humility) %>% 
    drop_na() %>%
    group_by(statement_id) %>% 
    summarise(humility = mean(humility))

# combine rating and response data
new_qlevel_results_weights_w_rating <- new_qlevel_results_weights %>%
    select(w3_qid, qid) %>%
    unique() %>%
    inner_join(rating_data_nona, by = c("w3_qid" = "statement_id")) %>% 
    ungroup()

####### Analyze ratings data: do edited texts move in the same direction as the original texts?

# Step 1: identify if texts are original or edits
new_qlevel_results_weights_w_rating <- new_qlevel_results_weights_w_rating %>% 
    mutate(original = str_detect(w3_qid, "_0$")) %>%
    # Step 1a: for each w3_qid, find average humility
    group_by(w3_qid) %>% 
    mutate(avg_humility = mean(humility)) %>% 
    ungroup()
    
# Step 2: for each edited text, find its original partner
original_texts <- new_qlevel_results_weights_w_rating %>% 
    filter(original) %>%
    select(qid, avg_humility) %>% 
    rename(original_humility = avg_humility) %>% 
    unique()
edited_texts <- new_qlevel_results_weights_w_rating %>% filter(!original) %>% 
    rename(edit_humility = humility) %>% 
    unique()
edited_original_pairs <- edited_texts %>% inner_join(original_texts, by = 'qid')

# Step 2a: find if the original text is supposed to be humble or not
edited_original_pairs <- edited_original_pairs %>%
    mutate(original_ih_status = str_detect(w3_qid, pattern = 'IHy'))

# Step 3: for each edited text, see if the humility of edited text and humility of original text moves in the proper direction
    # originally humble texts should become not humble (original_humble - humble >= 0)
    # originally not humble texts should become humble (original_humble - humble <= 0)
edited_original_pairs  <- edited_original_pairs %>% 
    mutate(humility_delta = original_humility - edit_humility) %>% 
    mutate(humility_delta_direction = ifelse(original_ih_status,
                                                    ifelse(humility_delta >= 0, "good", "bad"),
                                                    ifelse(humility_delta <= 0, "good", "bad")
      ))

# Step 4: plot the results. 

# Step 4a: pivot longer so that original and edit can show up together
edited_original_pairs_long <- edited_original_pairs %>%
    select(w3_qid, original_humility, edit_humility, humility_delta_direction) %>%
    pivot_longer(cols = c('original_humility', 'edit_humility')) %>% 
    separate(name, into = c('original_status', 'extra')) %>% 
    mutate(original_ih_status = ifelse(str_detect(w3_qid, pattern = 'IHy'), 'Original is IH', 'Original is Not IH')) %>% 
    mutate(original_status = factor(str_to_title(original_status), levels = c('Original', 'Edit'))) %>%
    mutate(alpha = ifelse(humility_delta_direction == 'bad', 1, 0.05))

png(paste0(directory, 'figures/ih_original_edit_slopes.png'), res = 300, height = 10, width = 12, units = 'in')
ggplot(edited_original_pairs_long) + 
    geom_line(aes(x = original_status, y = value, group = w3_qid, color = humility_delta_direction, alpha = alpha)) +
    facet_wrap(~original_ih_status) +
    theme_bw() +
    theme(legend.position = 'none') +
    xlab('Text Type') +
    ylab('Humility Rating') +
    theme(text = element_text(size = 20))
dev.off()


####### Remove any texts that do not move in the correct direction
keep_edit_qids <- edited_original_pairs %>% 
    filter(humility_delta_direction == 'good') %>% 
    pull(w3_qid)

edit_qlevel_keep <- new_qlevel_results_weights %>% 
    filter(w3_qid %in% keep_edit_qids)

original_qlevel <- new_qlevel_results_weights %>% 
    filter(str_detect(w3_qid, "_0$"))

qlevel_remove_bad_edits <- bind_rows(edit_qlevel_keep, original_qlevel)

## it's possible that some original texts have no remaining edits. remove such texts
keep_qids <- qlevel_remove_bad_edits %>% 
    # count number of posts per group and treatment status
    group_by(qid, ih_final) %>% 
    count() %>% 
    # count number of treatment statuses per group
    group_by(qid) %>% 
    count() %>% 
    # remove any groups with only 1 treatment status
    filter(n == 2) %>% 
    pull(qid)
qlevel_final <- qlevel_remove_bad_edits %>% filter(qid %in% keep_qids)

# isolate outcomes of interest
outcomes <- c("enjoy_num","persuasive_num","articulated","informative","aggressive","persuasive_me") %>% sort()

# intialize empty data structures
true_lower <- c()
true_upper <- c()
pt_est <- c()
models <- list()

# specify significance threshold
alpha <- 0.05

# for each outcome, regress outcome on treatment, fixed effect for original text group, and mixed effect for respondent
for(outcome in outcomes) {
        eqtn <- paste0(outcome, ' ~ ih_final + qid + (1 | ResponseId)')
        m <- lme4::lmer(eqtn,qlevel_final,weights = weight)
        models[[outcome]] <- m
        true_lower <- c(true_lower, summary(m)$coefficients['ih_finaly', 'Estimate'] - qnorm(1 - alpha/2) * summary(m)$coefficients['ih_finaly', 'Std. Error'])
        true_upper <- c(true_upper, summary(m)$coefficients['ih_finaly', 'Estimate'] + qnorm(1 - alpha/2) * summary(m)$coefficients['ih_finaly', 'Std. Error'])
        pt_est <- c(pt_est, summary(m)$coefficients['ih_finaly', 'Estimate'])

        print(paste(outcome, summary(m)$coefficients['ih_finaly', 'Estimate'], true_lower[length(true_lower)], true_upper[length(true_upper)], summary(m)$coefficients['ih_finaly', 'Std. Error']))
}


confint <- tibble(outcome_old = outcomes,
        lower = true_lower, 
        upper = true_upper,
        pt_est = pt_est
        ) %>%
        mutate(outcome = case_match(outcome_old,
                                    'persuasive_me' ~ 'persuades me',
                                    'persuasive_num' ~ 'persuades others',
                                    'enjoy_num' ~ 'enjoyable',
                                    .default = outcome_old
        ))


png(paste0(directory, '/figures/results_real_data_analysis.png'), res = 300, height = 10, width = 12, units = 'in')

ggplot(confint) +
geom_point(aes(x = outcome, y = pt_est), size = 6) +
geom_errorbar(aes(x = outcome, ymin = lower, ymax = upper), linewidth = 2) +
geom_abline(intercept = 0, slope = 0, linetype = 'dashed') +
theme_bw() +
xlab(NULL) +
ylab('ATE with 95% Conf Int') +
theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = 'none', text = element_text(size = 40)) 


dev.off()

## save regression results as a table
names(models) <- confint %>% arrange(outcome) %>% pull(outcome)

stargazer(models, type = "latex",
          title = "Regression Results for Real Data Analysis",
          align = TRUE,
          header = FALSE,
          font.size = "small",
          dep.var.labels.include = TRUE,
          dep.var.labels = names(models),
        #   dep.var.caption = "Dependent variable: MPG",
        #   model.names = TRUE,
        #   model.numbers = names(models),
          intercept.bottom = FALSE,
          intercept.top = TRUE,
        #   column.labels = names(models),
          omit.stat = c("f", "ser"),
          omit = 'qid',
          digits = 2,
          out = paste0(directory, 'real_data_analysis.tex'),
          label = 'tab:real_data_regs'
          )



### save original data frame with all variables after removing "bad" edits
new_qlevel_results_weights_full %>%
filter(row_id %in% qlevel_final$row_id) %>%
    write_csv(paste0(directory, 'qlevel_final.csv'))

