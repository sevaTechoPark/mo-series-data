import nest_asyncio
import asyncio

from run_bot import main_async

nest_asyncio.apply()

task = asyncio.create_task(main_async())