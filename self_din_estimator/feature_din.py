context_cat_col = ['same_city_look_order_start_city_code',
                   'same_city_look_order_passenger_count',
                   'order_name',
                   'hist_order_name'
                   # 'start_distance_split',
                   # 'price_split',
                   # 'time_diff_split'
                   ]
context_dense_col = [
    'same_city_look_order_price_scale_col',
    'same_city_look_order_start_distance_scale_col',
    'same_city_look_order_psg_distance_scale_col',
    'same_city_look_order_time_diff_scale_col',
    'same_city_look_order_start_time_diff_scale_col',
    'same_city_look_order_end_time_diff_scale_col',
    'same_city_look_order_time_diff_length_scale_col',
    'same_city_look_order_plog_publish_time_diff_scale_col',
    'same_city_look_order_plog_start_time_diff_scale_col',
    'same_city_look_order_pstart_publish_time_diff_scale_col',
    'same_city_look_order_start_rate_psg_scale_col'
]
driver_profile_cat_col = [
    'driver_sex',
    'driver_age',
    'driver_system_code',
    'driver_license_vehicle_type',
    'vehicle_brand_name',
    'rating',
    'driver_usual_route_lines',
    'vehicle_company',
    'vehicle_model_name',
    'apply_method'
]
driver_profile_dense_col = [
    'hbik_order_driver_cnt',
    'hbik_order_driver_accept_cnt',
    'driver_reply_days',
    'hbik_order_driver_distance',
    'hbik_order_driver_cnt_overcity',
    'hbik_order_driver_accept_cnt_overcity',
    'hbik_order_driver_finish_cnt_overcity',
    'hbik_order_driver_cnt_city',
    'hbik_order_driver_accept_cnt_city',
    'hbik_order_driver_finish_cnt_city',
    'hbik_order_driver_finish_cnt',
    'driver_finish_days',
    'hbik_order_driver_finish_app_cnt',
    'hbik_order_driver_sum_serviceprice',
    'hbik_order_driver_sum_bounty',
    'hbik_order_driver_sum_driverdiscount',
    'hbik_order_driver_sum_thanksfee',
    'hbik_order_driver_cnt_7d',
    'hbik_order_driver_finish_cnt_7d',
    'hbik_order_driver_finish_cnt_14d',
    'hbik_order_driver_finish_cnt_30d',
    'driver_plan_start_days',
    'hbik_driver_coupon_cnt_expire',
    'hbik_driver_coupon_cnt_used',
    'hbik_driver_coupon_amount_used',
    'hbik_driver_coupon_cnt_usable',
    'driver_bd_negative_comment_cnt',
    'driver_bd_negative_black_cnt',
    'driver_bd_negative_ticket_cnt',
    'driver_bd_negative_complain_cnt',
    'driver_zd_negative_comment_cnt',
    'driver_zd_negative_black_cnt',
    'driver_zd_negative_ticket_cnt',
    'driver_zd_negative_complain_cnt',
    'plan_start_rate',
    'driver_finish_rate',
    'reply_freq_avg',
    'finish_freq_avg',
    'driver_bd_negative_rate',
    'driver_zd_negative_rate',
    'hbik_order_driver_sum_fareprice',
    'hbik_order_driver_sum_driverpunishprice',
    'hbik_order_driver_finish_cnt_pre2w',
    'hbik_order_driver_finish_cnt_pre3w',
    'hbik_order_driver_finish_cnt_pre4w',
    'hbik_order_driver_finish_cnt_pre5w',
    'hbik_order_driver_finish_cnt_now_nature_month',
    'hbik_order_driver_finish_cnt_last_nature_month',
    'driving_experience',
    'hbik_view_driver_homepage_cnt_1d',
    'hbik_view_driver_lookorder_cnt_1d',
    'hbik_view_driver_choiceorder_cnt_1d',
    'hbik_view_driver_homepage_cnt_7d',
    'hbik_view_driver_lookorder_cnt_7d',
    'hbik_view_driver_choiceorder_cnt_7d',
    'hbik_view_driver_homepage_cnt_14d',
    'hbik_view_driver_lookorder_cnt_14d',
    'hbik_view_driver_choiceorder_cnt_14d',
    'hbik_view_driver_homepage_cnt_30d',
    'hbik_view_driver_lookorder_cnt_30d',
    'hbik_view_driver_choiceorder_cnt_30d'
]
psg_profile_cat_col = [
    'psg_sex',
    'psg_age'
]
psg_profile_dense_col = [
    'end_create_time_days',
    'hbik_order_passenger_cnt',
    'create_days',
    'reply_cnt',
    'psg_reply_days',
    'hbik_order_passenger_finish_cnt',
    'psg_finish_days',
    'hbik_order_passenger_cnt_overcity',
    'hbik_order_passenger_finish_cnt_overcity',
    'hbik_order_passenger_cnt_city',
    'hbik_order_passenger_finish_cnt_city',
    'hbik_order_passenger_sum_thanksfee',
    'hbik_order_passenger_sum_serviceprice',
    'hbik_order_passenger_finish_app_cnt',
    'hbik_order_passenger_cnt_7d',
    'hbik_order_passenger_deal_cnt_7d',
    'hbik_order_passenger_cnt_14d',
    'hbik_order_passenger_finish_cnt_14d',
    'hbik_order_passenger_cnt_30d',
    'hbik_order_passenger_deal_cnt_30d',
    'psg_plan_start_days',
    'hbik_passenger_coupon_cnt_usable',
    'hbik_passenger_coupon_cnt_common',
    'psg_bd_negative_comment_cnt',
    'psg_bd_negative_black_cnt',
    'psg_bd_negative_ticket_cnt',
    'psg_bd_negative_complain_cnt',
    'psg_zd_negative_comment_cnt',
    'psg_zd_negative_black_cnt',
    'psg_zd_negative_ticket_cnt',
    'psg_zd_negative_complain_cnt',
    'plan_start_days_rate',
    'reply_rate',
    'psg_finish_rate',
    'psg_bd_negative_rate',
    'psg_zd_negative_rate',
    'hbik_order_passenger_first_amount',
    'hbik_order_passenger_first_thanksfee',
    'hbik_order_passenger_first_serviceprice',
    'hbik_order_passenger_end_amount',
    'hbik_order_passenger_end_thanksfee',
    'hbik_order_passenger_end_serviceprice',
    'hbik_order_passenger_pay_cnt',
    'hbik_order_passenger_pay_cnt_city',
    'hbik_order_passenger_pay_cnt_overcity',
    'hbik_order_passenger_sum_amount',
    'exceed_cnt',
    'exceed_pay_cnt_180d',
    'hbik_order_passenger_nature_month_finish_cnt',
    'hbik_order_passenger_last_nature_month_finish_cnt',
    'hbik_order_passenger_finish_cnt_pre2w',
    'hbik_order_passenger_finish_cnt_pre3w',
    'hbik_order_passenger_finish_cnt_pre4w',
    'hbik_order_passenger_finish_cnt_pre5w',
    'hbik_passenger_invited_cnt',
    'hbik_order_passenger_after_accepted_canceled_cnt_1d',
    'hbik_order_passenger_before_accepted_cancel_cnt_1d',
    'hbik_order_passenger_after_accepted_cancel_cnt_1d',
    'hbik_order_passenger_finish_cnt_1w_weekend',
    'hbik_order_passenger_weekwork_cnt',
    'hbik_order_passenger_deal_weekwork_cnt'
]
print('all cols',
      len(context_cat_col) + len(context_dense_col) + len(driver_profile_cat_col) + len(
          driver_profile_dense_col) + len(psg_profile_dense_col) + len(psg_profile_cat_col))
