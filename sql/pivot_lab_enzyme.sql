with en_raw as (
select patientunitstayid, DATETIME_DIFF(charttime, intime, MINUTE) as chartoffset,
max(en.ast) as ast,
max(en.alt) as alt,
max(en.bilirubin_total) as bilirubin_total
from `cohort` cohort
join `physionet-data.mimic_derived.enzyme`  en
on cohort.subject_id = en.subject_id
AND cohort.hadm_id =en.hadm_id
where 
DATETIME_DIFF(starttime, en.charttime, MINUTE)<240
AND DATETIME_DIFF(endtime, en.charttime, MINUTE)>=0
AND  (ast is not null or alt is not null or bilirubin_total is not null)
group by patientunitstayid, chartoffset
order by patientunitstayid, chartoffset
)

select * from en_raw