import logging

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s :: [%(levelname)-8s] :: %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)