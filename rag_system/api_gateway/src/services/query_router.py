import logging
from dto.request import UserQuery
from dto.response import RoutingDecision
from integration.gemini_client import gemini_client

logger = logging.getLogger(__name__)

class QueryRouter:
    @staticmethod
    async def route(request: UserQuery) -> RoutingDecision:
        logger.info(f"Routing query: {request.query}")
        decision = await gemini_client.route_query(request.query)
        logger.info(f"Routing decision: {decision.query_type} - {decision.reasoning}")
        return decision
