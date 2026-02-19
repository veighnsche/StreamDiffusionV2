import { get, writable } from 'svelte/store';


export enum LCMLiveStatus {
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
    PAUSED = "paused",
    WAIT = "wait",
    SEND_FRAME = "send_frame",
    TIMEOUT = "timeout",
}

const initStatus: LCMLiveStatus = LCMLiveStatus.DISCONNECTED;

export const lcmLiveStatus = writable<LCMLiveStatus>(initStatus);
export const streamId = writable<string | null>(null);

let websocket: WebSocket | null = null;
let userId: string | null = null;

function generateUUIDv4(): string {
    const g = globalThis as typeof globalThis & {
        crypto?: Crypto;
        msCrypto?: Crypto;
    };
    const cryptoObj = g.crypto ?? g.msCrypto;

    if (cryptoObj && typeof cryptoObj.randomUUID === "function") {
        return cryptoObj.randomUUID();
    }

    const bytes = new Uint8Array(16);
    if (cryptoObj && typeof cryptoObj.getRandomValues === "function") {
        cryptoObj.getRandomValues(bytes);
    } else {
        for (let i = 0; i < bytes.length; i += 1) {
            bytes[i] = Math.floor(Math.random() * 256);
        }
    }

    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
}

export const lcmLiveActions = {
    async start(getSreamdata: () => any[]) {
        return new Promise((resolve, reject) => {

            try {
                // If an existing websocket exists and is open, reuse it
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                    websocket.send(JSON.stringify({ status: "resume", timestamp: Date.now() }));
                    streamId.set(userId);
                    resolve({ status: "connected"});
                } else {
                    websocket = null;
                }

                userId = generateUUIDv4();
                const websocketURL = `${window.location.protocol === "https:" ? "wss" : "ws"
                    }:${window.location.host}/api/ws/${userId}`;

                websocket = new WebSocket(websocketURL);
                websocket.onopen = () => {
                    console.log("Connected to websocket");
                };
                websocket.onclose = () => {
                    lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                    console.log("Disconnected from websocket");
                };
                websocket.onerror = (err) => {
                    console.error(err);
                };
                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    switch (data.status) {
                        case "connected":
                            lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                            streamId.set(userId);
                            resolve({ status: "connected", userId });
                            break;
                        case "send_frame":
                            if (get(lcmLiveStatus) === LCMLiveStatus.PAUSED) {
                                break;
                            }
                            lcmLiveStatus.set(LCMLiveStatus.SEND_FRAME);
                            const streamData = getSreamdata();
                            websocket?.send(JSON.stringify({ 
                                status: "next_frame", 
                                timestamp: Date.now()
                            }));
                            for (const d of streamData) {
                                this.send(d);
                            }
                            break;
                        case "wait":
                            if (get(lcmLiveStatus) === LCMLiveStatus.PAUSED) {
                                break;
                            }
                            lcmLiveStatus.set(LCMLiveStatus.WAIT);
                            break;
                        case "timeout":
                            console.log("timeout");
                            lcmLiveStatus.set(LCMLiveStatus.TIMEOUT);
                            streamId.set(null);
                            reject(new Error("timeout"));
                            break;
                        case "error":
                            console.log(data.message);
                            lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                            streamId.set(null);
                            reject(new Error(data.message));
                            break;
                    }
                };

            } catch (err) {
                console.error(err);
                lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                streamId.set(null);
                reject(err);
            }
        });
    },
    send(data: Blob | { [key: string]: any }) {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (data instanceof Blob) {
                websocket.send(data);
            } else {
                websocket.send(JSON.stringify(data));
            }
        } else {
            console.log("WebSocket not connected");
        }
    },
    async stop() {
        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
        if (websocket) {
            websocket.close();
        }
        websocket = null;
        streamId.set(null);
    },
    async pause() {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ status: "pause", timestamp: Date.now() }));
        }
        lcmLiveStatus.set(LCMLiveStatus.PAUSED);
        streamId.set(null);
    },
};
