with coa_raw as(
select patientunitstayid, DATETIME_DIFF(charttime, intime, MINUTE) as chartoffset,
max(coa.ptt) as ptt,
max(coa.pt) as pt,
max(coa.inr) as inr
from `cohort` cohort
left join `physionet-data.mimic_derived.coagulation`  coa
on cohort.subject_id = coa.subject_id
AND cohort.hadm_id =coa.hadm_id
where 
DATETIME_DIFF(starttime, coa.charttime, MINUTE)<240
AND DATETIME_DIFF(endtime, coa.charttime, MINUTE)>=0
AND  (ptt is not null or pt is not null or inr is not null)
group by patientunitstayid, chartoffset
order by patientunitstayid, chartoffset
)

select * from coa_raw