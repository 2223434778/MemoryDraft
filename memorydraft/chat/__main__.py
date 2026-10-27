from __future__ import annotations

import argparse
import ipaddress

from memorydraft.chat.config import ChatConfig


def main():
    parser = argparse.ArgumentParser(description="MemoryDraft friend chat gateway (one worker)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    config = ChatConfig.from_env()
    if not config.secure_cookie:
        address = ipaddress.ip_address(args.host)
        if not (address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")):
            parser.error("HTTP chat must bind to loopback or the server Tailscale IPv4")
    import uvicorn
    from memorydraft.chat.app import create_chat_app
    uvicorn.run(create_chat_app(config), host=args.host, port=args.port, workers=1,
                access_log=False, proxy_headers=False)


if __name__ == "__main__":
    main()
