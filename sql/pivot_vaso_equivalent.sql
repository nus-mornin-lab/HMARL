select co.stay_id, itemid, 
DATETIME_DIFF(starttime, intime, MINUTE) as start_chartoffset,
DATETIME_DIFF(endtime, intime, MINUTE) as end_chartoffset,
--extract( DATE from starttime) as starttime, extract (DATE from endtime) as endtime, -- rate, -- ,rateuom,
case when itemid in (30120,221906,30047) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3)  -- norad
when itemid in (30120,221906,30047) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3)  -- norad
when itemid in (30119,221289) and rateuom='mcg/kg/min' then round(cast(rate as numeric),3) -- epi
when itemid in (30119,221289) and rateuom='mcg/min' then round(cast(rate/80 as numeric),3) -- epi
when itemid in (30051,222315) and rate > 0.2 then round(cast(rate*5/60  as numeric),3) -- vasopressin, in U/h
when itemid in (30051,222315) and rateuom='units/min' then round(cast(rate*5 as numeric),3) -- vasopressin
when itemid in (30051,222315) and rateuom='units/hour' then round(cast(rate*5/60 as numeric),3) -- vasopressin
when itemid in (30128,221749,30127) and rateuom='mcg/kg/min' then round(cast(rate*0.45 as numeric),3) -- phenyl
when itemid in (30128,221749,30127) and rateuom='mcg/min' then round(cast(rate*0.45 / 80 as numeric),3) -- phenyl
when itemid in (221662,30043,30307) and rateuom='mcg/kg/min' then round(cast(rate*0.01 as numeric),3)  -- dopa
when itemid in (221662,30043,30307) and rateuom='mcg/min' then round(cast(rate*0.01/80 as numeric),3) else null end as rate_std-- dopa
from `upbeat-splicer-297704.sepsis_RL.cohort` co
left join `physionet-data.mimic_icu.inputevents` ie
on co.stay_id = ie.stay_id
where itemid in (30128,30120,30051,221749,221906,30119,30047,30127,221289,222315,221662,30043,30307) and rate is not null and statusdescription <> 'Rewritten'
order by stay_id, itemid, start_chartoffset