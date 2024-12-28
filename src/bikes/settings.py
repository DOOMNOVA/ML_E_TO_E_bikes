#Define settings for the application

import pydantic as pdt
import pydantic_settings as pdts

from bikes import jobs

class Settings(pdts.BaseSettings,strict=True,frozen=True,extra="forbid"):
    """Base class for application settings"""
    
    
    
    
    
class MainSettings(Settings):
    
    """Main settings of the application"""
    
    "Params - job(jobs.JobKind): job to run"
    
    job: jobs.JobKind = pdt.Field(...,discriminator="KIND")

