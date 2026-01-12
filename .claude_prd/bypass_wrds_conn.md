## Issue

WRDS blocks dynamic IP from connection

## Explore

Make plan to investigate whether update/app.py can bypass connection to WRDS. The classes that use WRDS are SecurityMaster, UniverseManager and CRSPDailyTicks. 

- CRSPDailyTicks is not used for updates. 
- UniverseManager should not be used in updates since symbols are fetched from `fetch_all_stocks`
- The hard part lies in bypassing `SecurityMaster`