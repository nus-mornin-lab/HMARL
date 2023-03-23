with
fluid0 as (
    select mv.stay_id, mv.itemid,
        DATETIME_DIFF(mv.storetime, ic.intime, MINUTE) AS chartoffset_fluid,
        round( case 
               when lower(mv.amountuom) = 'ml' then mv.amount
               else null
               end) as amount
    FROM `cohort`   co
    left JOIN `physionet-data.mimic_icu.inputevents` mv
    on co.patientunitstayid = mv.stay_id
      
    left join `physionet-data.mimic_icu.icustays` ic
    on co.patientunitstayid = ic.stay_id
    WHERE 
    
    (lower(amountuom) LIKE 'ml')
    AND mv.starttime > DATETIME_SUB(co.intime, INTERVAL 4 HOUR)
    AND mv.endtime < co.endtime
   -- GROUP BY stay_id, itemid,chartoffset_fluid
),

fluid1 as (
select stay_id, chartoffset_fluid, sum(amount) as chart_total 
from fluid0
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
)

select stay_id, chartoffset_fluid as chartoffset, 
max(intake_total) as intake_total, 
from fluid2
where chartoffset_fluid is not null
group by stay_id, chartoffset_fluid
order by stay_id, chartoffset_fluid