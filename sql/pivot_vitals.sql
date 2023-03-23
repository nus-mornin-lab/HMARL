select patientunitstayid,
DATETIME_DIFF(charttime, intime, MINUTE) as chartoffset,
heart_rate as heartrate,
sbp as ibp_systolic,
mbp as ibp_mean,
dbp as ibp_diastolic,
sbp_ni as nibp_systolic,
mbp_ni as nibp_mean,
dbp_ni as nibp_diastolic,
resp_rate as respiratoryrate,
spo2,
temperature

from `cohort` co
left join `physionet-data.mimic_derived.vitalsign`  vital
on co.patientunitstayid = vital.stay_id
where DATETIME_DIFF(starttime, charttime, MINUTE)<240
AND DATETIME_DIFF(endtime, charttime, MINUTE)>=0
AND (
heart_rate is not null 
or sbp is not null 
or dbp is not null 
or mbp is not null 
or dbp_ni is not null 
or sbp_ni is not null 
or mbp_ni is not null 
or temperature is not null 
or spo2 is not null
or resp_rate is not null
)
order by patientunitstayid, chartoffset

