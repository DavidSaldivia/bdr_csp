"""Test script for Chilean data loaders with Protocol-based design."""
from bdr_csp.pb_2 import (
    ModularCSPPlant,
    WeatherCL,
    MarketCL,
    WeatherMerra2,
    MarketAU,
    WeatherLoader,
    MarketLoader
)
from antupy import Var

def test_protocol_compliance():
    """Test that all loaders conform to protocols."""
    # Chilean loaders
    weather_cl = WeatherCL(
        lat=Var(-23.66, "deg"),
        lng=Var(-70.40, "deg"),
        year_i=2024,
        year_f=2024
    )
    market_cl = MarketCL(
        location="crucero",
        year_i=2024,
        year_f=2024
    )
    
    # Australian loaders
    weather_au = WeatherMerra2(
        lat=Var(-23., "deg"),
        lng=Var(115.9, "deg"),
        year_i=2019,
        year_f=2019
    )
    market_au = MarketAU(
        state="NSW",
        year_i=2019,
        year_f=2019
    )
    
    # Check protocol compliance
    assert isinstance(weather_cl, WeatherLoader), "WeatherCL must conform to WeatherLoader"
    assert isinstance(market_cl, MarketLoader), "MarketCL must conform to MarketLoader"
    assert isinstance(weather_au, WeatherLoader), "WeatherMerra2 must conform to WeatherLoader"
    assert isinstance(market_au, MarketLoader), "MarketAU must conform to MarketLoader"
    
    print("✓ All loaders conform to protocols")


def test_chilean_plant():
    """Test creating a plant with Chilean data loaders."""
    plant_cl = ModularCSPPlant(
        zf=Var(50, "m"),
        weather=WeatherCL(
            lat=Var(-23.66, "deg"),
            lng=Var(-70.40, "deg"),
            year_i=2024,
            year_f=2024,
            dT=0.5
        ),
        market=MarketCL(
            location="crucero",
            year_i=2024,
            year_f=2024,
            dT=0.5
        )
    )
    
    print(f"✓ Chilean plant created successfully")
    print(f"  Weather loader: {type(plant_cl.weather).__name__}")
    print(f"  Market loader: {type(plant_cl.market).__name__}")
    
    # Try loading data (will fail if files don't exist, but tests the interface)
    try:
        df_weather = plant_cl.weather.load_data()
        print(f"✓ Weather data loaded: {len(df_weather)} records")
        print(f"  Columns: {list(df_weather.columns)}")
        print(f"  Date range: {df_weather.index.min()} to {df_weather.index.max()}")
    except FileNotFoundError as e:
        print(f"⚠ Weather data not found (expected): {e}")
    
    try:
        df_market = plant_cl.market.load_data()
        print(f"✓ Market data loaded: {len(df_market)} records")
        print(f"  Columns: {list(df_market.columns)}")
    except FileNotFoundError as e:
        print(f"⚠ Market data not found (expected): {e}")


def test_australian_plant():
    """Test that Australian loaders still work."""
    plant_au = ModularCSPPlant(
        zf=Var(50, "m"),
        weather=WeatherMerra2(
            lat=Var(-23., "deg"),
            lng=Var(115.9, "deg"),
            year_i=2019,
            year_f=2019
        ),
        market=MarketAU(
            state="NSW",
            year_i=2019,
            year_f=2019
        )
    )
    
    print(f"✓ Australian plant created successfully")
    print(f"  Weather loader: {type(plant_au.weather).__name__}")
    print(f"  Market loader: {type(plant_au.market).__name__}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Protocol-based Data Loaders")
    print("=" * 60)
    print()
    
    test_protocol_compliance()
    print()
    
    test_chilean_plant()
    print()
    
    test_australian_plant()
    print()
    
    print("=" * 60)
    print("All tests passed! 🎉")
    print("=" * 60)
