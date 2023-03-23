with cohort as (
select co.*, ic.subject_id, ic.hadm_id,
 ic.outtime, ic.los as los_icu, ic.first_careunit
from `cohort` co
left join `physionet-data.mimic_icu.icustays`  ic
on co.patientunitstayid = ic.stay_id
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
)

select cohort.*, ad.ethnicity, ag.age,pa.gender, elix.elixhauser_vanwalraven, rrt.rrt_active as rrt_binary,we.weight as admissionweight, ht.height as admissionheight,oasis.oasis_score as first_oasis_score ,sofa.sofa_24hours as first_sofa_score, 
DATETIME_DIFF(ad.dischtime, ad.admittime, HOUR) as hosp_los_hours,
case when pa.dod is not null and DATETIME_DIFF(pa.dod,cohort.intime, DAY)<=28.0 then 1 else 0 end as mortality_28,
case when pa.dod is not null and DATETIME_DIFF(pa.dod, cohort.intime, DAY)<=90.0 then 1 else 0 end as mortality_90,
case when pa.dod is not null and DATETIME_DIFF(pa.dod, cohort.intime, DAY)<=0.0 then 1 else 0 end as mortality_icu,
ad.hospital_expire_flag as mortality_hospital
from cohort
left join `physionet-data.mimic_core.admissions`  ad
on cohort.subject_id = ad.subject_id 
and cohort.hadm_id = ad.hadm_id
left join `physionet-data.mimic_derived.age` ag
on cohort.subject_id = ag.subject_id 
and cohort.hadm_id = ag.hadm_id
left join  `physionet-data.mimic_core.patients` pa
on cohort.subject_id = pa.subject_id 
left join rrt
on cohort.patientunitstayid = rrt.stay_id
left join `physionet-data.mimic_derived.weight_durations` we
on cohort.patientunitstayid  = we.stay_id
left join oasis
on cohort.patientunitstayid  = oasis.stay_id
left join sofa
on cohort.patientunitstayid  = sofa.stay_id
left join `physionet-data.mimic_derived.height` as ht
on cohort.patientunitstayid = ht.stay_id
left join `elixhauser_quan` elix
on cohort.hadm_id = elix.hadm_id
where we.weight_type = 'admit'

order by subject_id, hadm_id, patientunitstayid
