"""
apiReportGenerator.py
"""
from openai import OpenAI
from pydantic import BaseModel
import instructor
from groq import Groq


import sys
from pathlib import Path
root_path = str(Path(__file__).absolute().parent.parent.parent.resolve())
app_path = str(Path(__file__).absolute().parent.parent.resolve())
sys.path.append(root_path)
sys.path.append(app_path) # add root and app project path to environment -> permits module import 

import logging
from app.conf.logManager import Logger
from conf.projectConfig import Config as cf
from mods.dataHandler import DataHandler
from projectSetup import Setup


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
      pass

   def get_traffic_context(self, n_reports):
      initial_prompt = " Create an event report answering the questions What, When, Why, How, Who and Where. \
      The event is a car accident during jam hours. Structure the report according to the 6 questions. Be as compact as possible. \
      The answer to each question is the following: What: a car accident; When: Tuesday at 8 pm; Where: in Bordeaux city in the A63 highway; \
      Why: Car A driver was looking at his phone; Who: two drivers that came out of work; How: car A was driving in the highway on the right lane, car B was in front and started to brake rapidly because there was a traffic jam coming, car A driver did not see car B approaching and did not brake soon enough.\
      Create some extra information about the details. \
      Modify the structure, instead of diving the sections this way, do it as a unique report following the chain of events."

      previous_message = "Reference Report: High-Traffic Collision on Bordeaux’s A63 Highway \
      On Tuesday at 8 PM, during peak evening traffic, a rear-end collision occurred on the A63 highway in Bordeaux, involving two commuters returning home from work. The incident unfolded as follows: \
      Vehicule B, traveling in the right lane, abruptly braked upon encountering a sudden traffic jam near downtown Bordeaux. However, the driver of Vehicule A, trailing behind, failed to react in time. Investigations later revealed that Car A’s operator had been distracted by their phone, delaying critical braking. The impact caused moderate damage to both vehicles and minor injuries to the drivers, though neither required hospitalization. \
      Emergency services arrived promptly, managing congestion and clearing debris. Traffic was temporarily rerouted, exacerbating delays during the already busy hour. Witnesses highlighted the stop-and-go flow preceding the accident, consistent with post-work congestion patterns. Authorities reiterated warnings against mobile phone use while driving, citing this incident as a preventable example of distracted driving. \
      The collision underscored the risks of high-density traffic scenarios, where split-second attention deficits can escalate rapidly."

      input_prompt = "create " + str(n_reports) + " variations of the report with different traffic accident situations, locations, consequences,\
      evidence supporting the events and vehicules involved. Do not label the sections of each variation. \
      The reports must have a similar length of the given Reference Report. \
      In the reports there must always be a vehicule A and a vehicule B. You can introduce more vehicules but always respecting the order A,B,C,D,... You can define the vehicle type: truck, car, motorcycle, van, etc."

      messages=[
              {"role": "system", "content": "You are a reporting agent assistant. \
                                              You create car accident reports in English. \
                                              You answer the what, when, why, who, how and where questions about the events. \
                                              You give also extra information about the contingency actions. \
                                              You extract all the event pipinformation into the given structure: what, when, why, who, how, where, contingency actions and event report."},
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
    
     input_prompt = "create " + str(n_reports) + " cases. You can vary the context of the deviations to multiple areas in pharma industry (such as manufacturing, quality control, documentation, raw materials, personnel, shipping, storage, stability studies, etc.).\
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
   
   # Generate N reports
   def generate_reports(self, n_reports: int, report_id: int, use_chat_gpt_api = True):
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
  
      dh.export_to_excel_from_api_response(report_data, model_name, filename)


if __name__ == "__main__":  
    from apiReportGenerator import ApiReportGenerator, ApiReport, ApiReports
    from conf.projectConfig import Config as cf

    api = ApiReportGenerator()
    N_REPORTS = 5
    api.generate_reports(n_reports=N_REPORTS,
                        report_id=cf.API.TRAFFIC_REPORT_ID,
                        use_chat_gpt_api=True)