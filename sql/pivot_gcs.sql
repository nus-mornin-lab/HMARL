with vital1 as (
select co.stay_id,
DATETIME_DIFF(charttime, intime, MINUTE) as chartoffset,
gcs
from `cohort` co
left join `physionet-data.mimic_derived.gcs` gcs_score
on co.stay_id = gcs_score.stay_id
order by stay_id, chartoffset
)
select * from vital1