"""
Cognee Knowledge Graph Pipeline for Taxi Trip Data Analysis
"""

# Standard library imports
import os
import asyncio

# Third-party imports
from dotenv import load_dotenv
import dlt
import requests
import pandas as pd

# Cognee imports
import cognee
from cognee import config
from cognee.modules.engine.models import NodeSet
from cognee.modules.search.types import SearchType
from cognee.api.v1.visualize.visualize import visualize_graph


# =============================================================================
# Configuration
# =============================================================================

def setup_environment():
    """Setup environment variables and configuration."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["LLM_API_KEY"] = api_key
    os.environ["GRAPH_DATABASE_PROVIDER"] = "kuzu"
    config.set_llm_api_key(api_key)


# =============================================================================
# Data Pipeline (DLT)
# =============================================================================

@dlt.resource(write_disposition="replace", name="zoomcamp_data")
def zoomcamp_data():
    """Extract and process taxi trip data."""
    url = "https://us-central1-dlthub-analytics.cloudfunctions.net/data_engineering_zoomcamp_api"
    response = requests.get(url)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['Trip_Pickup_DateTime'] = pd.to_datetime(df['Trip_Pickup_DateTime'])

    # Define time buckets
    df['tag'] = pd.cut(
        df['Trip_Pickup_DateTime'],
        bins=[
            pd.Timestamp("2009-06-01"),
            pd.Timestamp("2009-06-10"),
            pd.Timestamp("2009-06-20"),
            pd.Timestamp("2009-06-30")
        ],
        labels=["first_10_days", "second_10_days", "last_10_days"],
        right=False
    )

    # Drop rows not in the specified range
    df = df[df['tag'].notnull()]
    yield df


def run_dlt_pipeline():
    """Create and run the DLT pipeline."""
    pipeline = dlt.pipeline(
        pipeline_name="zoomcamp_pipeline",
        destination="duckdb",
        dataset_name="zoomcamp_tagged_data"
    )
    pipeline.run(zoomcamp_data())
    dataset = pipeline.dataset().zoomcamp_data.df()
    return dataset


# =============================================================================
# Knowledge Graph (Cognee)
# =============================================================================

async def build_knowledge_graph(dataset):
    """Build knowledge graph using Cognee."""
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Add data for each time period
    time_periods = ["first_10_days", "second_10_days", "last_10_days"]

    for period in time_periods:
        df_subset = dataset.loc[dataset["tag"] == period].copy()
        df_subset.drop(columns=["tag"], inplace=True)
        data_json = df_subset.to_json(orient="records", lines=False)
        await cognee.add(data_json, node_set=[period])

    await cognee.cognify()


async def search_cognee(query, node_set, query_type=SearchType.GRAPH_COMPLETION):
    """Search the knowledge graph."""
    answer = await cognee.search(
        query_text=query,
        query_type=query_type,
        node_type=NodeSet,
        node_name=node_set,
        top_k=5
    )
    return answer


# =============================================================================
# Main Execution
# =============================================================================

async def main():
    """Main pipeline execution."""
    setup_environment()

    # Run DLT pipeline to load data into DuckDB
    dataset = run_dlt_pipeline()

    # Build knowledge graph
    await build_knowledge_graph(dataset)

    # Generate visualization
    visualization_path = "./graph_visualization.html"
    await visualize_graph(visualization_path)


async def search_example():
    """Example search function."""
    setup_environment()

    results = await search_cognee(
        "What's in this knowledge graph?",
        node_set=["first_10_days"]
    )

    print(results[0])


if __name__ == "__main__":
    # Run main pipeline
    asyncio.run(main())

    # Run search example - In case you already have your data stored locally,
    # comment out the previous line of code to avoid running though LLM again

    asyncio.run(search_example())