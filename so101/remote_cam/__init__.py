"""Watch and control the lab cameras from outside the home network.

The HP Omen publisher is left unchanged. A process on the Mac subscribes to
its existing ZMQ stream and either serves it directly or pushes it out to a
relay that phones and laptops can open from any network.
"""
