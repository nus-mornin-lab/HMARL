with le_raw as(
select patientunitstayid, DATETIME_DIFF(charttime, intime, MINUTE) as chartoffset,
max(le.wbc) as wbc,
max(le.platelet ) as platelet
from `cohort` cohort
left join `physionet-data.mimic_derived.complete_blood_count` le
on cohort.subject_id = le.subject_id
AND cohort.hadm_id =le.hadm_id 
where 
DATETIME_DIFF(starttime, le.charttime, MINUTE)<240
AND DATETIME_DIFF(endtime, le.charttime, MINUTE)>=0
AND  (wbc is not null or platelet is not null)
group by patientunitstayid, chartoffset
order by patientunitstayid, chartoffset
)

select * from le_raw