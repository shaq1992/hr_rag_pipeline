from pydantic import BaseModel

class UserQuery(BaseModel):
    query: str
    user_id: str = "default_user"
