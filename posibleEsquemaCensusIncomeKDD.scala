val censusSchema = StructType(Array(StructField("age", IntegerType, true),
    StructField("class_of_worker", StringType, true),
    StructField("detailed_industry_recode", StringType, true),
    StructField("detailed_occupation_recode", StringType, true),
    StructField("education", StringType, true),
    StructField("wage_per_hour", IntegerType, true),
    StructField("enroll_in_edu_inst_last_wk", StringType, true),
    StructField("marital_stat", StringType, true),
    StructField("major_industry_code", StringType, true),
    StructField("major_occupation_code", StringType, true),
    StructField("race", StringType, true),
    StructField("hispanic_origin", StringType, true),
    StructField("sex", StringType, true),
    StructField("member_of_a_labor_union", StringType, true),
    StructField("reason_for_unemployment", StringType, true),
    StructField("full_or_part_time_employment_stat", StringType, true),
    StructField("capital_gains", IntegerType, true),
    StructField("capital_losses", IntegerType, true),
    StructField("dividends_from_stocks", IntegerType, true),
    StructField("tax_filer_stat", StringType, true),
    StructField("region_of_previous_residence", StringType, true),
    StructField("state_of_previous_residence", StringType, true),
    StructField("detailed_household_and_family_stat", StringType, true),
    StructField("detailed_household_summary_in_household", StringType, true),
    StructField("instance_weight", DoubleType, true),
    StructField("migration_code-change_in_msa", StringType, true),
    StructField("migration_code-change_in_reg", StringType, true),
    StructField("migration_code-move_within_reg", StringType, true),
    StructField("live_in_this_house_1_year_ago", StringType, true),
    StructField("migration_prev_res_in_sunbelt", StringType, true),
    StructField("num_persons_worked_for_employer", IntegerType, true),
    StructField("family_members_under_18", StringType, true),
    StructField("country_of_birth_father", StringType, true),
    StructField("country_of_birth_mother", StringType, true),
    StructField("country_of_birth_self", StringType, true),
    StructField("citizenship", StringType, true),
    StructField("own_business_or_self_employed", StringType, true),
    StructField("fill_inc_questionnaire_for_veterans_admin", StringType, true),
    StructField("veterans_benefits", StringType, true),
    StructField("weeks_worked_in_year", IntegerType, true),
    StructField("year", IntegerType, true),
    StructField("_instance_weight", StringType, true)))
