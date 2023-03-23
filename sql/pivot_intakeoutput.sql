WITH urine AS
(
    SELECT  co.patientunitstayid as stay_id,
    DATETIME_DIFF(uo.charttime, co.intime, MINUTE) AS chartoffset_uo,
    urineoutput
            

    FROM `cohort`   co
    LEFT JOIN `physionet-data.mimic_derived.urine_output` uo 
    ON co.patientunitstayid = uo.stay_id
   
),

urine1 as (
select stay_id, chartoffset_uo, urineoutput,
(select
sum(urineoutput)
from urine r1
where r1.stay_id = r0.stay_id
AND r1.chartoffset_uo <= r0.chartoffset_uo
) as output_total
from urine r0
order by stay_id,chartoffset_uo
),

fluid0 as (
    select mv.stay_id, mv.itemid,
        DATETIME_DIFF(mv.starttime, co.intime, MINUTE) AS chartoffset_fluid,
        round( case when mv.amountuom = 'L' then mv.amount * 1000.0
               when mv.amountuom = 'ml' then mv.amount
               else null
               end) as amount
    FROM `upbeat-splicer-297704.MV_mimic.cohort_final`   co
    left JOIN `physionet-data.mimic_icu.inputevents` mv
    on co.patientunitstayid = mv.stay_id
    WHERE mv.itemid IN (225944,
        220949,
        220952,220950,
        225158,
        225943,226089,
        225828,227533,
        225159,
        225823,225825,225827,225941,225823,
        225161,
        220995,
        220862,220864,
        225916,225917,225948,225947,
        225920,
        225969)
    AND amountuom LIKE 'ml' OR amountuom LIKE 'L'
   -- GROUP BY stay_id, itemid,chartoffset_fluid
),

fluid1 as (
select stay_id, chartoffset_fluid, sum(amount) as chart_total from fluid0
group by stay_id, chartoffset_fluid
order by stay_id, chartoffset_fluid
),
fluid2 as (
select stay_id, chartoffset_fluid,chart_total,
(select sum(chart_total) 
from fluid1 temp2
where temp1.stay_id = temp2.stay_id
AND temp2.chartoffset_fluid<=temp1.chartoffset_fluid) as intake_total
from fluid1 temp1
order by stay_id,chartoffset_fluid
),

with intake_output_raw as (
select urine1.stay_id, chartoffset_fluid, chartoffset_uo,intake_total, output_total
from urine1 left join fluid2 on urine1.stay_id = fluid2.stay_id
order by stay_id, chartoffset_fluid, chartoffset_uo
),

temp as(
select stay_id, chartoffset_fluid, intake_total,
case when chartoffset_fluid>=chartoffset_uo then output_total else 0 end as output_total_1
from intake_output_raw
order by stay_id, chartoffset_fluid,intake_total
),
temp1 as (
select stay_id, chartoffset_fluid, intake_total,
max (output_total_1) as output_total
from temp
group by stay_id, chartoffset_fluid, intake_total
order by stay_id, chartoffset_fluid, intake_total
)
select stay_id as patientunitstayid, chartoffset_fluid as chartoffset, intake_total, 
output_total,intake_total-output_total as nettotal from temp1
where chartoffset_fluid is not null
order by stay_id, chartoffset_fluid


