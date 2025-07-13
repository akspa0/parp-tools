# TODO

Implement through the api from wago.tools, a way to grab pm4, pd4, wmo, and other related assets, so users don't have to go searching for it from unreputable sources.

@https://wago.tools/apis has a guide on how to request data properly. We will gather a list of known-good data, and implement a way to request it from the api.

I'm copying and pasting the data from that page here, for safe keeping.

API Endpoints

    https://wago.tools/api/builds
        Lists every version available for the API
    https://wago.tools/api/builds/latest
        Lists the latest version for each available product
    https://wago.tools/api/builds/{product}/latest
        Gets a singular latest version for a specific product
        Where {product} is a WoW product
    https://wago.tools/api/casc/{fdid}
    Supports version
        Gets a CASC file based on the FDID.
    https://wago.tools/api/info/{fdid}
        Get information about a file based on FDID
    https://wago.tools/api/files
    Supports version
    Supports format
        Gets all files available in a specific version

Flags
Version:

    Supports version argument, when passing in a full build, e.g. 10.1.0.50000.
    Supports product argument, when passing in a product, e.g. wow_classic_beta, providing the latest version.

Format:

    Supports format argument, providing either csv (default) or json.

------

## TODO 2

We also need to look at lib\wow.tools.local
It's mostly c# code and is continuously updated, and may have loaders or exporters for the files we are parsing. It also has handling for all database-type files, like dbc, db2, wdb, adb. This could be useful for decoding the raw data into our Core.v2 library. It would be ideal to reference their code and not implement our own janky implementation from it. I'm not certain there's a 3d viewer in the library yet, but there is someone that has built such a tool for a web experience (javascript or nodejs or something - I just have to find the repo!)
