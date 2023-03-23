WITH co AS
(
  select ih.stay_id, ie.hadm_id
  , hr
  -- start/endtime can be used to filter to values within this hour
  , DATETIME_SUB(ih.endtime, INTERVAL '1' HOUR) AS starttime
  , ih.endtime
  from `physionet-data.mimic_derived.icustay_hourly` ih
  INNER JOIN `physionet-data.mimic_icu.icustays` ie
    ON ih.stay_id = ie.stay_id
),
vs as (
  select co.stay_id, co.hr
  -- vitals
  , min(v.temperature) as temperature_min
  , max(v.temperature) as temperature_max
  , max(v.heart_rate) as heart_rate_max
  , max(v.resp_rate) as resp_rate_max
  from co
  left join `physionet-data.mimic_derived.vitalsign` v
    on co.stay_id = v.stay_id
    and co.starttime < v.charttime
    and co.endtime >= v.charttime
  group by co.stay_id, co.hr

),

bg as (
  select co.stay_id, co.hr
  -- blood gas
  , min(bgas.pco2) as paco2_min
  from co
  left join `physionet-data.mimic_derived.bg` bgas
    on co.hadm_id = bgas.hadm_id
    and co.starttime < bgas.charttime
    and co.endtime >= bgas.charttime
  group by co.stay_id, co.hr

),

lab as (
  select co.stay_id, co.hr
  -- blood gas
  , min(lb.wbc) as wbc_min
  , max(lb.wbc) as wbc_max
  from co
  left join `physionet-data.mimic_derived.complete_blood_count` lb
    on co.hadm_id = lb.hadm_id
    and co.starttime < lb.charttime
    and co.endtime >= lb.charttime
  group by co.stay_id, co.hr

),

bd as (
  select co.stay_id, co.hr
  -- blood gas
  , max(blood_dif.bands) as bands_max
  from co
  left join `physionet-data.mimic_derived.blood_differential` blood_dif
    on co.hadm_id = blood_dif.hadm_id
    and co.starttime < blood_dif.charttime
    and co.endtime >= blood_dif.charttime
  group by co.stay_id, co.hr

),


scorecomp as
(
select co.stay_id
  , co.hr
  , co.starttime, co.endtime
  , vs.temperature_min
  , vs.temperature_max
  , vs.heart_rate_max
  , vs.resp_rate_max
  , bg.paco2_min
  , lab.wbc_min
  , lab.wbc_max
  , bd.bands_max
FROM co
left join bg
 on co.stay_id = bg.stay_id
 and co.hr = bg.hr
left join vs
  on co.stay_id = vs.stay_id
  and co.hr = vs.hr
left join lab
  on co.stay_id = lab.stay_id
  and co.hr = lab.hr
left join bd
  on co.stay_id = bd.stay_id
  and co.hr = bd.hr
),
scorecalc as
(
  -- Calculate the final score
  -- note that if the underlying data is missing, the component is null
  -- eventually these are treated as 0 (normal), but knowing when data is missing is useful for debugging
  select scorecomp.*

  , case
      when temperature_min < 36.0 then 1
      when temperature_max > 38.0 then 1
      when temperature_min is null then null
      else 0
    end as temp_score


  , case
      when heart_rate_max > 90.0  then 1
      when heart_rate_max is null then null
      else 0
    end as heart_rate_score

  , case
      when resp_rate_max > 20.0  then 1
      when paco2_min < 32.0  then 1
      when coalesce(resp_rate_max, paco2_min) is null then null
      else 0
    end as resp_score

  , case
      when wbc_min <  4.0  then 1
      when wbc_max > 12.0  then 1
      when bands_max > 10 then 1-- > 10% immature neurophils (band forms)
      when coalesce(wbc_min, bands_max) is null then null
      else 0
    end as wbc_score

  from scorecomp
),

score_final as
(
  select s.*
    -- Combine all the scores to get SIRS
    -- Impute 0 if the score is missing
   -- the window function takes the max over the last 24 hours
    , coalesce(
        MAX(temp_score) OVER (PARTITION BY stay_id ORDER BY HR
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
      ,0) as temp_score_24hours
     , coalesce(
         MAX(heart_rate_score) OVER (PARTITION BY stay_id ORDER BY hr
         ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
        ,0) as heart_rate_score_24hours
    , coalesce(
        MAX(resp_score) OVER (PARTITION BY stay_id ORDER BY HR
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
      ,0) as resp_score_24hours
    , coalesce(
        MAX(wbc_score) OVER (PARTITION BY stay_id ORDER BY HR
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
      ,0) as wbc_score_24hours

    -- sum together data for final SOFA
    , coalesce(
        MAX(temp_score) OVER (PARTITION BY stay_id ORDER BY HR
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
      ,0)
     + coalesce(
         MAX(heart_rate_score) OVER (PARTITION BY stay_id ORDER BY HR
         ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
      ,0)
     + coalesce(
        MAX(resp_score) OVER (PARTITION BY stay_id ORDER BY HR
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
      ,0)
     + coalesce(
        MAX(wbc_score) OVER (PARTITION BY stay_id ORDER BY HR
        ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING)
      ,0)

    as sirs_24hours
  from scorecalc s
  WINDOW W as
  (
    PARTITION BY stay_id
    ORDER BY hr
    ROWS BETWEEN 23 PRECEDING AND 0 FOLLOWING
  )
)
select * from score_final
where hr >= 0;