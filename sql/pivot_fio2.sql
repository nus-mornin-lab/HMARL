
with cohort as (
select co.*, ic. subject_id, ic.hadm_id  
from `cohort` co
left join `physionet-data.mimic_icu.icustays` ic
on co.patientunitstayid = ic.stay_id
)

select patientunitstayid,
DATETIME_DIFF(charttime, intime, MINUTE) as chartoffset,
max( tidal_volume_observed ) as Tidal_volume_1,
max( tidal_volume_set ) as Tidal_volume_2,
max( tidal_volume_spontaneous ) as Tidal_volume_3,
max( peep ) as PEEP_1,
max( fio2 ) as FiO2_1,
from cohort
left join `physionet-data.mimic_derived.ventilator_setting`   venti
on cohort.subject_id = venti.subject_id
AND cohort.patientunitstayid =venti.stay_id 
where 
DATETIME_DIFF(starttime, venti.charttime, MINUTE)<240
AND DATETIME_DIFF(endtime, venti.charttime, MINUTE)>=0
AND (
tidal_volume_set is not null 
or tidal_volume_observed is not null 
or tidal_volume_spontaneous is not null 
or peep is not null 
or fio2 is not null 
)
group by patientunitstayid, chartoffset
order by patientunitstayid, chartoffset

