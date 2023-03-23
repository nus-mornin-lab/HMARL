with ch_raw as (
select co.patientunitstayid, DATETIME_DIFF(charttime, intime, MINUTE) as chartoffset,
case when ce.itemid = 220635 then valuenum else null end as magnesium,
case when ce.itemid = 225667 then valuenum else null end as ionized_calcium,
case when ce.itemid = 225625 then valuenum else null end as calcium,
case when ce.itemid = 228640 then valuenum else null end as etco2,
case when ce.itemid = 227442 then valuenum else null end as potassium, 
case when ce.itemid = 220645 or ce.itemid = 228389 then valuenum else null end as sodium, 
case when ce.itemid = 220621 then valuenum else null end as glucose, 
case when ce.itemid = 220228 then valuenum else null end as hemoglobin,
case when ce.itemid = 227443 then valuenum else null end as bicarbonate

from `cohort` co
left join `physionet-data.mimic_icu.chartevents` ce
on co.subject_id = ce.subject_id
AND co.hadm_id =ce.hadm_id
-- group by stay_id, chartoffset
order by stay_id, chartoffset
),

ch as (
select patientunitstayid, chartoffset,
max(magnesium) as magnesium,
max(ionized_calcium) as ionized_calcium,
max(calcium) as calcium,
max(etco2) as etco2,
max(potassium) as potassium,
max(sodium) as sodium,
max(glucose) as glucose,
max(hemoglobin) as hemoglobin,
max(bicarbonate) as bicarbonate
from ch_raw
group by patientunitstayid,chartoffset
order by patientunitstayid,chartoffset
)

select *,
from ch
where magnesium is not null
or ionized_calcium is not null
or calcium is not null
or etco2 is not null
or potassium is not null
or sodium is not null
or glucose is not null
or hemoglobin is not null
or bicarbonate is not null
order by patientunitstayid, chartoffset

