with ch_raw as (
select patientunitstayid, DATETIME_DIFF(charttime, intime, MINUTE) as chartoffset,
max(ch.bun ) as bun,
max(ch.albumin) as albumin,
max(ch.creatinine ) as creatinine,
from `cohort` cohort
left join `physionet-data.mimic_derived.chemistry`  ch
on cohort.subject_id = ch.subject_id
AND cohort.hadm_id =ch.hadm_id
where 
DATETIME_DIFF(starttime, ch.charttime, MINUTE)<240
AND DATETIME_DIFF(endtime, ch.charttime, MINUTE)>=0
AND (bun is not null or albumin is not null or creatinine is not null)
group by patientunitstayid, chartoffset
order by patientunitstayid, chartoffset
)
select * from ch_raw