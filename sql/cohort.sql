
with sep3 as (
select sep.subject_id, sep.stay_id, ic.hadm_id , suspected_infection_time,sofa_time, ic.intime, ic.los as los_icu, ic.first_careunit
from `physionet-data.mimic_derived.sepsis3` sep
left join `physionet-data.mimic_icu.icustays`  ic
on sep.subject_id =ic.subject_id 
and sep.stay_id = ic.stay_id
where sep.sepsis3 is TRUE
order by subject_id,stay_id
),

rank_ICU as (
     SELECT  subject_id
                , hadm_id as hadmid
                , stay_id
                , intime, outtime
                , row_number() OVER (PARTITION BY subject_id ORDER BY hadm_id) AS first_hosp
                , row_number() OVER (PARTITION BY subject_id, hadm_id ORDER BY intime) AS first_icu
               
       FROM `physionet-data.mimic_icu.icustays`  
       ),
       
rrt as (
select temp1.stay_id,
max(dialysis_active) as rrt_active
from `physionet-data.mimic_derived.rrt` temp1
left join `physionet-data.mimic_icu.icustays` temp2
on temp1.stay_id=temp2.stay_id 
where DATETIME_DIFF(temp1.charttime, temp2.intime, HOUR)<72.0
group by stay_id
order by stay_id),

oasis as (
select stay_id, max(oasis) as oasis_score
from `physionet-data.mimic_derived.oasis` 
group by stay_id
order by stay_id
),

sofa_raw as (
select stay_id, sofa_24hours,
row_number() OVER (PARTITION BY stay_id ORDER BY starttime asc) AS first_sofa
from `physionet-data.mimic_derived.sofa` as sofa 
),

sofa as (
select * from sofa_raw
where first_sofa=1
order by stay_id
),

vaso_raw as (
select sep3.stay_id,
DATETIME_DIFF(vaso.starttime, intime, MINUTE) as chartoffset,
case
when itemid = 221289 then "epinephrine"
when itemid = 221906 then "dopamine"
when itemid = 221906 then "norepinephrine"
when itemid = 221749 then "phenylephrine"
when itemid = 222315 then "vasopressin"
else null end as drugname,
case when itemid in (221289, -- epinephrine
                     221662, -- dopamine
                     221906, -- norepinephrine
                     221749, -- phenylephrine
                     222315) -- vasopressin
                      

    and amount is not null
    then amount else null end as dose,
    amountuom,
from sep3
left join `physionet-data.mimic_icu.inputevents` vaso
on sep3.subject_id = vaso.subject_id
AND sep3.stay_id =vaso.stay_id 
where DATETIME_DIFF(vaso.starttime, sep3.intime,HOUR)<=72
order by stay_id, chartoffset
),

vasopressor as (
select stay_id, 
case when max(dose) is not null then 1 else 0 end as vaso_binary
from vaso_raw
group by stay_id
order by stay_id
)

select sep3.*,r.first_hosp, r.first_icu, ad.ethnicity, ag.age,pa.gender, rrt.rrt_active as rrt_binary,we.weight,oasis.oasis_score as first_oasis_score ,sofa.sofa_24hours as first_sofa_score, va.vaso_binary,
DATETIME_DIFF(ad.dischtime, ad.admittime, HOUR) as hosp_los_hours,
case when pa.dod is not null and DATETIME_DIFF(pa.dod,ad.dischtime, DAY)<=28.0 then 1 else 0 end as mortality_28,
case when pa.dod is not null and DATETIME_DIFF(pa.dod, ad.dischtime, DAY)<=90.0 then 1 else 0 end as mortality_90,
case when pa.dod is not null and DATETIME_DIFF(pa.dod, r.outtime, DAY)<=0.0 then 1 else 0 end as mortality_icu,
ad.hospital_expire_flag as mortality_hospital
from sep3
left join rank_ICU r
on sep3.subject_id = r.subject_id 
and sep3.hadm_id = r.hadmid 
and sep3.stay_id = r.stay_id 
left join `physionet-data.mimic_core.admissions`  ad
on sep3.subject_id = ad.subject_id 
and sep3.hadm_id = ad.hadm_id
left join `physionet-data.mimic_derived.age` ag
on sep3.subject_id = ag.subject_id 
and sep3.hadm_id = ag.hadm_id
left join  `physionet-data.mimic_core.patients` pa
on sep3.subject_id = pa.subject_id 
left join rrt
on sep3.stay_id = rrt.stay_id
left join `physionet-data.mimic_derived.weight_durations` we
on sep3.stay_id = we.stay_id
left join oasis
on sep3.stay_id = oasis.stay_id
left join sofa
on sep3.stay_id = sofa.stay_id
left join vasopressor va
on sep3.stay_id =va.stay_id
where we.weight_type = 'admit'
order by subject_id, hadm_id, stay_id
