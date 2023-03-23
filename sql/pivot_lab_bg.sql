with bg_raw as (
select cohort.patientunitstayid,
DATETIME_DIFF(bg.charttime, intime, MINUTE) as chartoffset,
max(bg.ph) as pH,
max(bg.po2) as pao2,
max(bg.pco2) as paco2,
max(bg.bicarbonate) as bicarbonate,
max(bg.lactate) as lactate,
max(bg.hemoglobin ) as hemoglobin,
max(bg.baseexcess) as baseexcess,
max(bg.chloride) as chloride,
max(bg.glucose) as glucose,
max(bg.calcium) as calcium,
max(bg.potassium) as potassium,
max(bg.sodium) as sodium,
max(bg.totalco2 ) as co2,
max(bg.pao2fio2ratio ) as pao2fio2ratio
from `cohort` cohort
left join `physionet-data.mimic_derived.bg`  bg
on cohort.subject_id = bg.subject_id
AND cohort.hadm_id =bg.hadm_id 
where 
DATETIME_DIFF(starttime, bg.charttime, MINUTE)<240
AND DATETIME_DIFF(endtime, bg.charttime, MINUTE)>=0
AND  
(
ph is not null 
or po2 is not null 
or pco2 is not null 
or bicarbonate is not null 
or lactate is not null 
or baseexcess is not null
or chloride is not null 
or glucose is not null 
or calcium is not null
or potassium is not null
or sodium is not null
or pao2fio2ratio is not null
)
group by patientunitstayid, chartoffset
order by patientunitstayid, chartoffset
)

select *
from bg_raw
