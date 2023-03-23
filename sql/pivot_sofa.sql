select patientunitstayid,
DATETIME_DIFF(sofa_score.starttime, intime, MINUTE) as chartoffset,
sofa_score.sofa_24hours
from `cohort` co
left join `physionet-data.mimic_derived.sofa` sofa_score
on co.patientunitstayid = sofa_score.stay_id
where DATETIME_DIFF(co.starttime, sofa_score.starttime, MINUTE)<240
AND DATETIME_DIFF(co.endtime, sofa_score.endtime, MINUTE)>=0
order by patientunitstayid, chartoffset

