# load model if does exists. This is a hack. Poetry does not include static files yet
# Script will move the base_model.pkl and stops.pkl from env file to current location

from pathlib import Path

store_model = Path.cwd() / 'hisia/models/base_model.pkl'
store_stops = Path.cwd() / 'hisia/data/stops.pkl'

if not store_model.exists():
    from shutil import move
    import hisia
     
    base = Path(hisia.__file__).parent
    store_model.parent.mkdir(parents=True, exist_ok=True)
    store_stops.parent.mkdir(parents=True, exist_ok=True)

    locate_model = base / 'models/base_model.pkl'
    locate_stops =base / 'data/stops.pkl'

    move(locate_model, store_model)  
    move(locate_stops, store_stops)
  

from .hisia import Hisia