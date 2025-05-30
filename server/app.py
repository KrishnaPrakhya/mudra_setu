from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import List



app=FastAPI()


class Person(BaseModel):
  id:int
  name:str
  price:float

db:List[Person]=[]

@app.post("/api/person",response_model=Person)
def post_person(person:Person):

  db.append()

