"""
apiReportGenerator.py

Generates synthetic reports using LLM APIs (GROQ, ChatGPT)

"""
import sys
from pathlib import Path
root_path = str(Path(__file__).absolute().parent.parent.parent.resolve())
app_path = str(Path(__file__).absolute().parent.parent.resolve())
sys.path.append(root_path)
sys.path.append(app_path) 

from openai import OpenAI
from pydantic import BaseModel
import instructor
from groq import Groq

import logging
from app.conf.logManager import Logger
from conf.projectConfig import Config as cf
from mods.dataHandler import DataHandler
from projectSetup import Setup
from datetime import datetime as dt

logging.setLoggerClass(Logger)
log = logging.getLogger(__name__)
env_dict = Setup().config  # API keys saved in environment variables
dh = DataHandler()

# Nested structure
class ApiReport(BaseModel):
    report_name: str
    what: str
    when: str
    where: str
    who: str
    how: str
    why: str
    contingency_actions: str
    report: str

class ApiReports(BaseModel):
  reports: list[ApiReport]


class ApiReportGenerator():
   
   def __init__(self):
      self.call_id = None
    
   def init_id(self, report_id: str):
      if report_id == cf.API.TRAFFIC_REPORT_ID:
         self.call_id = "API_TRAFFIC_" + dt.now().strftime("%d%m%Y%H%M%S")
      elif report_id == cf.API.PHARMA_REPORT_ID:
         self.call_id = "API_PHARMA_" + dt.now().strftime("%d%m%Y%H%M%S")

   def get_traffic_context(self, n_reports):
      initial_prompt = "Generate a report of a mock traffic accident with less than 700 characters. \
      From the report, extract the following key information:\
     - what: a synthetic keyword description of the event that happend.\
     - when: date and time of the event discovery or occurrence. Both can be reported, but it has to be clear whether they refer to the discovery or occurrence of the event.\
     - where: a synthetic keyword description of location where the event has took place or was discovered\
     - who: a synthetic keyword description of name and job position of each involved person in the event and their role in the event (e.g. who was responsible of the event's occurrence?...)\
     - how: a synthetic keyword description of how did the event happened. As root cause may not be identified yet, the how could be unknown.\
     - why: a synthetic keyword description of the cause of the event. As root cause may not be identified yet, the why could be unknown.\
     - contingency_actions: a synthetic keyword description of all actions that have been immediately performed to contain the problem.\
     It is important to focus on information accuracy: any information reported in the text description of the accident must be mentioned in the key information details." \
     "Identify vehicules involved with the letters A, B, C in order of appearance"
    
      previous_message = """REFERENCE REPORT: On August 24, 2025, at 17:40, Vehicle A (a city bus) driven by Mr. Roberto 
      Mena collided with Vehicle B (a motorcycle) operated by Ms. Irene Valdés
        on Calle de Alcalá, Madrid. The bus was turning right when the motorcycle attempted to overtake from the right-hand 
        side, causing a side impact. The incident was reported at 17:45 by a pedestrian. Ms. Valdés sustained minor injuries. 
        Police and EMT arrived promptly. Area was secured and traffic was redirected.
      - when: Occurrence: August 24, 2025, 17:40; Discovery: August 24, 2025, 17:45
      - where: Calle de Alcalá, Madrid
      - who: Mr. Roberto Mena – A city bus (Vehicle A); Ms. Irene Valdés, injured – a motorcycle (Vehicle B)
      - how: Motorcycle attempted right-side overtake as bus turned
      - why: Risky overtaking maneuver by Vehicle B
      - contingency_actions: Police and EMT response, traffic redirection, area secured
      """

      input_prompt = "Create " + str(n_reports) + " coherent traffic accident reports, each report has > 300 characters long, "
      "with the same structure as the REFERENCE REPORT and followed by key information extracted from the report."
      "Each report must tag the vehicles involved 'Vehicle {LETTER}' with its driver; e.g. : Sarah Kim reversed her car (vehicle A). Order by the vehicle letters as the entities appear in the chain of the events LETTER=A,B,C,D... "
      "Use a neutral, factual tone."
      "Rules: \n"
      "- Always make sure that any information in the report is reflected in the key information sections (what, when, where, who, how, why and contingency_actions) and vice versa. \n"
      "- Vary context (location, time, road type, weather, consequences, actions) as provided. \n"
      "- Do not create ficticious information that is not contained in the key information sections \n"
      "- Ensure that all 'Vehicle {LETTER}' tags appearing in the who information also appear in the report. \n"
      "- Time may include both occurrence and discovery when provided."

      messages=[
              {"role": "system", "content": "You are a reporting agent assistant. \
                                              You create car accident reports in English. \
                                              You answer the what, when, why, who, how and where questions about the events. \
                                              You give also extra information about the contingency actions. \
                                              You extract all the event information into the given structure: report title, what, when, why, who, how, where, contingency actions and event report."},
              {"role": "user", "content": initial_prompt},
              {"role": "assistant", "content": previous_message},  # Include the initial response
              {"role": "user", "content": input_prompt}  # New follow-up prompt
      ]
      return messages

   def get_pharma_context(n_reports):
     initial_prompt = "Generate a description text of a mock deviation occurring in a pharmaceutical manufacturing process. \
     From the generated text, extract the following key information:\
     - what: a synthetic keyword description of the event that happend. If a product or batch number is mentioned in the text description, it must be reported in the what section.\
     - when: date and time of the event discovery or occurrence. Both can be reported, but it has to be clear whether they refer to the discovery or occurrence of the event.\
     - where: a synthetic keyword description of location where the event has took place or was discovered\
     - who: a synthetic keyword description of name and job position of each involved person in the event and their role in the event (e.g. who reported or discovered the event? who was responsible of the event's occurrence?...)\
     - how: a synthetic keyword description of how did the event happened. As root cause may not be identified yet, the how could be unknown.\
     - why: a synthetic keyword description of the cause of the event. As root cause may not be identified yet, the why could be unknown.\
     - contingency_actions: a synthetic keyword description of all actions that have been immediately performed to contain the problem. If root cause investigation has been started, mention it along with the deviation ID if available.\
     It is important to focus on information accuracy: any information reported in the text description of the deviation must be mentioned in the key information details."
    
     previous_message = "Reference Deviation Description: On July 16, 2025 at 14:35, a deviation occurred during the sterile filtration step of Batch 5H2A-PEN in Cleanroom C-102 (Sterile Suite). The power supply to the filtration skid unexpectedly shut down mid-process, causing an unplanned interruption of product flow for approximately 6 minutes. The event was discovered immediately by Elena Morales (Process Technician), who was monitoring the operation at the time. She promptly informed Michael Chen (Shift Supervisor) and the Engineering team.\
     Initial investigation indicates that a power distribution panel overload may have caused the interruption, though root cause is still under investigation. The power supply was restored by Engineering (David Liu, Facilities Engineer) within 10 minutes of the incident. A decision was made to quarantine the impacted batch until product integrity could be assessed.\
     Contingency actions included immediate process hold, batch quarantine, environmental monitoring in the affected area, and initiation of a formal deviation investigation (Ref: DEV-2025-1093). No personnel were harmed, and no breach of sterility has been observed so far.\
     - what: power interruption during sterile filtration of batch 5H2A-PEN\
     - when: occurrence and discovery : July 16, 2025 at 14:35\
     - where: Cleanroom C-102 (Sterile Suite)\
     - who: Elena Morales, Process Technician – discovered and reported the event; Michael Chen, Shift Supervisor – informed and coordinated response; David Liu, Facilities Engineer – restored power\
     - how: unexpected shutdown of filtration skid due to suspected power distribution panel overload, causing ~6-minute process interruption\
     - why: unknown – root cause under investigation\
     - contingency_actions: process hold, batch quarantine, environmental monitoring, power restoration, deviation investigation initiated (Ref: DEV-2025-1093)"
    
     input_prompt = "Create " + str(n_reports) + " cases. You can vary the context of the deviations to multiple areas in pharma industry (such as manufacturing, quality control, documentation, raw materials, personnel, shipping, storage, stability studies, etc.).\
     Do not label the sections of each variation. \
     Always make sure that any information in the deviation report is reflected in the key information sections (what, when, where, who, how, why and contingency_actions) and vice versa.\
     The reports must have a similar length to the given Reference Deviation Description."
    
     messages=[
             {"role": "system", "content": "You are a reporting agent assistant. \
                                             You create pharmaceutical industry deviation reports in English. \
                                             You answer the what, when, where, who, how, and why questions about the events. \
                                             You give also extra information about the contingency_actions. \
                                             You extract all the event information into the given structure: what, when, where, who, how, why, contingency_actions and deviation_description."},
             {"role": "user", "content": initial_prompt},
             {"role": "assistant", "content": previous_message},  # Include the initial response
             {"role": "user", "content": input_prompt}  # New follow-up prompt
     ]
     return messages
    
   #Chat GPT API
   def __call_gpt_api(self, messages, model):        
        # ChatGPT (pay)
        # "gpt-4.1-mini"
        # "gpt-4.1" # Careful it is expensive
        # "gpt-4.1-nano" # Does not work well for car A and B instructions
      client = OpenAI(api_key=env_dict["API_GPT_KEY"])
      model_name = model
  
      response = client.responses.parse(
          model = model_name,
          input = messages,
          text_format=ApiReports,
      )
      return response.output_parsed.reports, model_name
   
   # Open Router
   def __call_open_router_api(messages):
      open_router_model = "google/gemma-3-27b-it:free"        
      client = OpenAI(
         base_url="https://openrouter.ai/api/v1", 
         api_key=env_dict["API_OPEN_ROUTER_KEY"],)        # Use instructor for handling structured outputs
      client = instructor.from_openai(client,  mode=instructor.Mode.JSON)        
      model_name = open_router_model        
      completion = client.chat.completions.create(
          model=model_name,
          messages= messages,
          response_model=ApiReports,
      )
      return completion.reports, model_name
   
    # GROQ
   def __call_groq_api(messages, model):
      model_name = model
      client = Groq(
          api_key=env_dict["API_GROQ_KEY"],
      )
      client = instructor.from_groq(client, mode=instructor.Mode.JSON)  
      chat_completion = client.chat.completions.create(
          messages=messages,
          response_model=ApiReports,
          model=model_name,
      )  
      report_data = chat_completion.reports
      return report_data, model_name
   
   def generate_n_reports(self, 
                          total_n_reports: int, 
                          batch_size: int, 
                          report_id: int, 
                          use_chat_gpt_api = True):
      """
      Automatically generate N reports from groq or GPT API in chunks of given size.
      The batch size must be a divisor of total_n_reports.
      """
      self.init_id(report_id=report_id)
      for i in range(int(total_n_reports/batch_size)):
        print(f"******* Generating BATCH {i+1} of {int(total_n_reports/batch_size)} *******")
        self._generate_chunk_n_reports(n_reports=batch_size,
                                      report_id=report_id,
                                      use_chat_gpt_api=use_chat_gpt_api)          
   # Generate report by chunks of N reports
   def _generate_chunk_n_reports(self, n_reports: int, report_id: int, use_chat_gpt_api = True):
      """
      Automatically generate N reports from groq or GPT API.
      If GPT API not selected, then GROQ API is selected by default
      """
      if report_id == cf.API.TRAFFIC_REPORT_ID:
        messages = self.get_traffic_context(n_reports=n_reports)
        filename = "traffic_reports"
      elif report_id == cf.API.PHARMA_REPORT_ID:
        messages = self.get_pharma_context(n_reports=n_reports)
        filename = "pharma_reports"
      
      log.info("Sending API call to model...retrieving reports...")
      if use_chat_gpt_api:
        report_data, model_name = self.__call_gpt_api(messages, model=cf.API.GPT_API_MODEL)
      else:
      # report_data, model_name = call_open_router_api(messages)
        report_data, model_name = self.__call_groq_api(messages, model=cf.API.GROQ_API_MODEL)
  
      dh.export_to_excel_from_api_response(report_data, model_name, filename, sheet_name=self.call_id)


if __name__ == "__main__": 

  from apiReportGenerator import ApiReportGenerator, ApiReport, ApiReports
  from conf.projectConfig import Config as cf

  api = ApiReportGenerator()
  api.generate_n_reports(total_n_reports=200, 
                          batch_size=10, 
                          report_id=cf.API.TRAFFIC_REPORT_ID, 
                          use_chat_gpt_api=True)