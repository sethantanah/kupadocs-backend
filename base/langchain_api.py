
import os, json, settings, openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser



openai.api_key = settings.OPENAI_API_KEY
llm_model = settings.OPENAI_DEFAULT_LLM
chat = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY,temperature=0.0, model=llm_model)


resume_template = """\
For the following text, extract the most relevant information.\
Note: keep the same structure, if information is not availale in the content leave blank values.\

Format the output as JSON with the following keys:
name
contact_information
work_experience
education
skills
certifications
languages
additional_information

text: {text}
"""



def get_resume_data(resume_text:str):
    """### Parse the LLM output string into a Python dictionary"""

    name_schema = ResponseSchema(name="name",
                                description="name of person")

    contact_schema = ResponseSchema(name="contact_information",
                                description='{\
                                            "address": "123 Main Street, City, State, Zip",\
                                            "phone": "123-456-7890",\
                                            "email": "johndoe@example.com"\
                                            }')


    work_experiance_schema = ResponseSchema(name="work_experience",
                                description='[{\
                                            "title": "Job Title",\
                                            "employer": "Company Name",\
                                            "location": "City, State",\
                                            "dates": "Start Date - End Date",\
                                            "description": "Brief description of responsibilities and achievements."\
                                            }]')


    education_schema = ResponseSchema(name="education",
                                description='[{\
                                            "degree": "Degree Obtained",\
                                            "major": "Major",\
                                            "institution": "University Name",\
                                            "location": "City, State",\
                                            "dates": "Graduation Date"\
                                            }]')


    skills_schema = ResponseSchema(name="skills",
                                description='[\
                                            "Skill 1",\
                                            "Skill 2",\
                                            "Skill 3"\
                                        ]')


    certifications_schema = ResponseSchema(name="certifications",
                                description='[\
                                            "Certification 1",\
                                            "Certification 2"\
                                        ]')


    languages_schema = ResponseSchema(name="languages",
                                description='[\
                                            "Language 1",\
                                            "Language 2"\
                                        ]')


    additional_information_schema = ResponseSchema(name="additional_information",
                                description='Any additional information or achievements.')

    response_schemas = [name_schema, contact_schema, work_experiance_schema, education_schema,
                        skills_schema, certifications_schema, languages_schema, additional_information_schema]


    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(template=resume_template)

    messages = prompt.format_messages(text=resume_text,
                                    format_instructions=format_instructions)

    response = chat(messages)
    output_dict = {}

    if response:
       try:
          output_dict = output_parser.parse(response.content)
       except Exception as e:
        output_dict = json.loads(response.content)

    return output_dict

    
    