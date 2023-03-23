with eliflg as
(
select hadm_id, seq_num, icd_code
, CASE
  when icd_code in ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493') then 1
  when SUBSTR(icd_code, 1, 4) in ('4254','4255','4257','4258','4259') then 1
  when SUBSTR(icd_code, 1, 3) in ('428','I50') then 1
  else 0 end as chf       /* Congestive heart failure */

, CASE
  when icd_code in ('42613','42610','42612','99601','99604') then 1
  when SUBSTR(icd_code, 1, 4) in ('4260','4267','4269','4270','4271','4272','4273','4274','4276','4278','4279','7850','V450','V533') then 1
  else 0 end as arrhy

, CASE
  when SUBSTR(icd_code, 1, 4) in ('0932','7463','7464','7465','7466','V422','V433') then 1
  when SUBSTR(icd_code, 1, 3) in ('394','395','396','397','424','I34') then 1
  else 0 end as valve     /* Valvular disease */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('4150','4151','4170','4178','4179','I289') then 1
  when SUBSTR(icd_code, 1, 3) in ('416') then 1
  else 0 end as pulmcirc  /* Pulmonary circulation disorder */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('0930','4373','4431','4432','4438','4439','4471','5571','5579','V434','I739') then 1
  when SUBSTR(icd_code, 1, 3) in ('440','441') then 1
  else 0 end as perivasc  /* Peripheral vascular disorder */

, CASE
  when SUBSTR(icd_code, 1, 3) in ('401','I10') then 1
  else 0 end as htn       /* Hypertension, uncomplicated */

, CASE
  when SUBSTR(icd_code, 1, 3) in ('402','403','404','405') then 1
  else 0 end as htncx     /* Hypertension, complicated */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('3341','3440','3441','3442','3443','3444','3445','3446','3449','G839') then 1
  when SUBSTR(icd_code, 1, 3) in ('342','343') then 1
  else 0 end as para      /* Paralysis */

, CASE
  when icd_code in ('33392') then 1
  when SUBSTR(icd_code, 1, 4) in ('3319','3320','3321','3334','3335','3362','3481','3483','7803','7843') then 1
  when SUBSTR(icd_code, 1, 3) in ('334','335','340','341','345') then 1
  else 0 end as neuro     /* Other neurological */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('4168','4169','5064','5081','5088','J449') then 1
  when SUBSTR(icd_code, 1, 3) in ('490','491','492','493','494','495','496','500','501','502','503','504','505') then 1
  else 0 end as chrnlung  /* Chronic pulmonary disease */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2500','2501','2502','2503','E089','E099', 'E139') then 1
  else 0 end as dm        /* Diabetes w/o chronic complications*/

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2504','2505','2506','2507','2508','2509') then 1
  else 0 end as dmcx      /* Diabetes w/ chronic complications */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2409','2461','2468','E039') then 1
  when SUBSTR(icd_code, 1, 3) in ('243','244') then 1
  else 0 end as hypothy   /* Hypothyroidism */

, CASE
  when icd_code in ('40301','40311','40391','40402','40403','40412','40413','40492','40493') then 1
  when SUBSTR(icd_code, 1, 4) in ('5880','V420','V451','I179') then 1
  when SUBSTR(icd_code, 1, 3) in ('585','586','V56') then 1
  else 0 end as renlfail  /* Renal failure */

, CASE
  when icd_code in ('07022','07023','07032','07033','07044','07054') then 1
  when SUBSTR(icd_code, 1, 4) in ('0706','0709','4560','4561','4562','5722','5723','5724','5728','5733','5734','5738','5739','V427','K769') then 1
  when SUBSTR(icd_code, 1, 3) in ('570','571') then 1
  else 0 end as liver     /* Liver disease */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('5317','5319','5327','5329','5337','5339','5347','5349','K274') then 1
  else 0 end as ulcer     /* Chronic Peptic ulcer disease (includes bleeding only if obstruction is also present) */

, CASE
  when SUBSTR(icd_code, 1, 3) in ('042','043','044','B20','Z21','R75') then 1
  else 0 end as aids      /* HIV and AIDS */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2030','2386','C859') then 1
  when SUBSTR(icd_code, 1, 3) in ('200','201','202') then 1
  else 0 end as lymph     /* Lymphoma */

, CASE
  when SUBSTR(icd_code, 1, 3) in ('196','197','198','199') then 1
  when SUBSTR(icd_code, 1, 4) in ('C799') then 1
  else 0 end as mets      /* Metastatic cancer */

, CASE
  when SUBSTR(icd_code, 1, 3) in
  (
     '140','141','142','143','144','145','146','147','148','149','150','151','152'
    ,'153','154','155','156','157','158','159','160','161','162','163','164','165'
    ,'166','167','168','169','170','171','172','174','175','176','177','178','179'
    ,'180','181','182','183','184','185','186','187','188','189','190','191','192'
    ,'193','194','195'
  ) then 1
  when SUBSTR(icd_code, 1, 4) in ('C801') then 1
  else 0 end as tumor     /* Solid tumor without metastasis */

, CASE
  when icd_code in ('72889','72930') then 1
  when SUBSTR(icd_code, 1, 4) in ('7010','7100','7101','7102','7103','7104','7108','7109','7112','7193','7285','3571') then 1
  when SUBSTR(icd_code, 1, 3) in ('446','714','720','725') then 1
  else 0 end as arth              /* Rheumatoid arthritis/collagen vascular diseases */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2871','2873','2874','2875','D689') then 1
  when SUBSTR(icd_code, 1, 3) in ('286') then 1
  else 0 end as coag      /* Coagulation deficiency */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2780') then 1
  when SUBSTR(icd_code, 1, 3) in ('E66') then 1
  else 0 end as obese     /* Obesity      */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('7832','7994','R634') then 1
  when SUBSTR(icd_code, 1, 3) in ('260','261','262','263') then 1
  else 0 end as wghtloss  /* Weight loss */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2536','E878') then 1
  when SUBSTR(icd_code, 1, 3) in ('276') then 1
  else 0 end as lytes     /* Fluid and electrolyte disorders */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2800') then 1
  when SUBSTR(icd_code, 1, 3) in ('D62') then 1
  else 0 end as bldloss   /* Blood loss anemia */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2801','2808','2809','D509') then 1
  when SUBSTR(icd_code, 1, 3) in ('281') then 1
  else 0 end as anemdef  /* Deficiency anemias */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2652','2911','2912','2913','2915','2918','2919','3030','3039','3050','3575','4255','5353','5710','5711','5712','5713','V113','F101') then 1
  when SUBSTR(icd_code, 1, 3) in ('980') then 1
  else 0 end as alcohol /* Alcohol abuse */

, CASE
  when icd_code in ('V6542') then 1
  when SUBSTR(icd_code, 1, 4) in ('3052','3053','3054','3055','3056','3057','3058','3059') then 1
  when SUBSTR(icd_code, 1, 3) in ('292','304','F19') then 1
  else 0 end as drug /* Drug abuse */

, CASE
  when icd_code in ('29604','29614','29644','29654') then 1
  when SUBSTR(icd_code, 1, 4) in ('2938') then 1
  when SUBSTR(icd_code, 1, 3) in ('295','297','298','F29') then 1
  else 0 end as psych /* Psychoses */

, CASE
  when SUBSTR(icd_code, 1, 4) in ('2962','2963','2965','3004') then 1
  when SUBSTR(icd_code, 1, 3) in ('309','311','F32','F33') then 1
  else 0 end as depress  /* Depression */
from `physionet-data.mimic_hosp.diagnoses_icd` icd
where seq_num != 1 -- we do not include the primary icd-9 code
)
-- collapse the icd9_code specific flags into hadm_id specific flags
-- this groups comorbidities together for a single patient admission
, eligrp as
(
  select hadm_id
  , max(chf) as chf
  , max(arrhy) as arrhy
  , max(valve) as valve
  , max(pulmcirc) as pulmcirc
  , max(perivasc) as perivasc
  , max(htn) as htn
  , max(htncx) as htncx
  , max(para) as para
  , max(neuro) as neuro
  , max(chrnlung) as chrnlung
  , max(dm) as dm
  , max(dmcx) as dmcx
  , max(hypothy) as hypothy
  , max(renlfail) as renlfail
  , max(liver) as liver
  , max(ulcer) as ulcer
  , max(aids) as aids
  , max(lymph) as lymph
  , max(mets) as mets
  , max(tumor) as tumor
  , max(arth) as arth
  , max(coag) as coag
  , max(obese) as obese
  , max(wghtloss) as wghtloss
  , max(lytes) as lytes
  , max(bldloss) as bldloss
  , max(anemdef) as anemdef
  , max(alcohol) as alcohol
  , max(drug) as drug
  , max(psych) as psych
  , max(depress) as depress
from eliflg
group by hadm_id
),
-- now merge these flags together to define elixhauser
-- most are straightforward.. but hypertension flags are a bit more complicated

elixhauser_quan as (
select adm.hadm_id
, chf as congestive_heart_failure
, arrhy as cardiac_arrhythmias
, valve as valvular_disease
, pulmcirc as pulmonary_circulation
, perivasc as peripheral_vascular
-- we combine "htn" and "htncx" into "HYPERTENSION"
, case
    when htn = 1 then 1
    when htncx = 1 then 1
  else 0 end as hypertension
, para as paralysis
, neuro as other_neurological
, chrnlung as chronic_pulmonary
-- only the more severe comorbidity (complicated diabetes) is kept
, case
    when dmcx = 1 then 0
    when dm = 1 then 1
  else 0 end as diabetes_uncomplicated
, dmcx as diabetes_complicated
, hypothy as hypothyroidism
, renlfail as renal_failure
, liver as liver_disease
, ulcer as peptic_ulcer
, aids as aids
, lymph as lymphoma
, mets as metastatic_cancer
-- only the more severe comorbidity (metastatic cancer) is kept
, case
    when mets = 1 then 0
    when tumor = 1 then 1
  else 0 end as solid_tumor
, arth as rheumatoid_arthritis
, coag as coagulopathy
, obese as obesity
, wghtloss as weight_loss
, lytes as fluid_electrolyte
, bldloss as blood_loss_anemia
, anemdef as deficiency_anemias
, alcohol as alcohol_abuse
, drug as drug_abuse
, psych as psychoses
, depress as depression

FROM `physionet-data.mimic_core.admissions` adm
left join eligrp eli
  on adm.hadm_id = eli.hadm_id
order by adm.hadm_id
)

select hadm_id
,  -- Below is the van Walraven score
   0 * aids +
   0 * alcohol_abuse +
  -2 * blood_loss_anemia +
   7 * congestive_heart_failure +
   -- Cardiac arrhythmias are not included in van Walraven based on Quan 2007
   3 * chronic_pulmonary +
   3 * coagulopathy +
  -2 * deficiency_anemias +
  -3 * depression +
   0 * diabetes_complicated +
   0 * diabetes_uncomplicated +
  -7 * drug_abuse +
   5 * fluid_electrolyte +
   0 * hypertension +
   0 * hypothyroidism +
   11 * liver_disease +
   9 * lymphoma +
   12 * metastatic_cancer +
   6 * other_neurological +
  -4 * obesity +
   7 * paralysis +
   2 * peripheral_vascular +
   0 * peptic_ulcer +
   0 * psychoses +
   4 * pulmonary_circulation +
   0 * rheumatoid_arthritis +
   5 * renal_failure +
   4 * solid_tumor +
  -1 * valvular_disease +
   6 * weight_loss
as elixhauser_vanwalraven



,  -- Below is the 29 component SID score
   0 * aids +
  -2 * alcohol_abuse +
  -2 * blood_loss_anemia +
   -- Cardiac arrhythmias are not included in SID-29
   9 * congestive_heart_failure +
   3 * chronic_pulmonary +
   9 * coagulopathy +
   0 * deficiency_anemias +
  -4 * depression +
   0 * diabetes_complicated +
  -1 * diabetes_uncomplicated +
  -8 * drug_abuse +
   9 * fluid_electrolyte +
  -1 * hypertension +
   0 * hypothyroidism +
   5 * liver_disease +
   6 * lymphoma +
   13 * metastatic_cancer +
   4 * other_neurological +
  -4 * obesity +
   3 * paralysis +
   0 * peptic_ulcer +
   4 * peripheral_vascular +
  -4 * psychoses +
   5 * pulmonary_circulation +
   6 * renal_failure +
   0 * rheumatoid_arthritis +
   8 * solid_tumor +
   0 * valvular_disease +
   8 * weight_loss
as elixhauser_SID29


,  -- Below is the 30 component SID score
   0 * aids +
   0 * alcohol_abuse +
  -3 * blood_loss_anemia +
   8 * cardiac_arrhythmias +
   9 * congestive_heart_failure +
   3 * chronic_pulmonary +
  12 * coagulopathy +
   0 * deficiency_anemias +
  -5 * depression +
   1 * diabetes_complicated +
   0 * diabetes_uncomplicated +
 -11 * drug_abuse +
  11 * fluid_electrolyte +
  -2 * hypertension +
   0 * hypothyroidism +
   7 * liver_disease +
   8 * lymphoma +
  17 * metastatic_cancer +
   5 * other_neurological +
  -5 * obesity +
   4 * paralysis +
   0 * peptic_ulcer +
   4 * peripheral_vascular +
  -6 * psychoses +
   5 * pulmonary_circulation +
   7 * renal_failure +
   0 * rheumatoid_arthritis +
  10 * solid_tumor +
   0 * valvular_disease +
  10 * weight_loss
as elixhauser_SID30

from  elixhauser_quan


