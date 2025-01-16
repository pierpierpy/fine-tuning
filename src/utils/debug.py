import debugpy


def listen(host: str, port: int):
    try:
        debugpy.listen((host, port))

        print(f"Listening on {host}:{port}")
    except:
        listen(host, port + 1)
        print("------------")
