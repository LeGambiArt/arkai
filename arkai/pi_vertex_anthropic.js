import { createRequire } from "node:module";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const PROVIDER_ID = "google-vertex-anthropic";
const ADC_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
const VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16";
const DEFAULT_PI_PACKAGE_JSON = "__ARKAI_PI_PACKAGE_JSON__";

function getProject() {
    return process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT || process.env.GOOGLE_VERTEX_PROJECT;
}

function getLocation() {
    return process.env.GOOGLE_CLOUD_LOCATION || process.env.GOOGLE_VERTEX_LOCATION || process.env.VERTEX_LOCATION;
}

function findPiPackageJson() {
    const starts = [
        process.env.ARKAI_PI_PACKAGE_JSON,
        DEFAULT_PI_PACKAGE_JSON,
        process.env.PI_PACKAGE_DIR,
        process.argv[1],
    ];
    for (const start of starts) {
        if (!start) continue;
        let current;
        try {
            current = realpathSync(start);
        } catch {
            current = resolve(start);
        }
        if (!existsSync(current) || basename(current) !== "package.json") {
            current = dirname(current);
        }
        while (true) {
            const candidate =
                basename(current) === "package.json" ? current : join(current, "package.json");
            try {
                const packageData = JSON.parse(readFileSync(candidate, "utf8"));
                if (packageData.name === "@earendil-works/pi-coding-agent") return candidate;
            } catch {
                // Continue walking toward the package root.
            }
            const parent = dirname(current);
            if (parent === current) break;
            current = parent;
        }
    }
    throw new Error("Arkai could not locate Pi's package for the Vertex Anthropic provider");
}

function getVertexUrl(project, location, model) {
    const host = location === "global" ? "aiplatform.googleapis.com" : `${location}-aiplatform.googleapis.com`;
    const encodedProject = encodeURIComponent(project);
    const encodedLocation = encodeURIComponent(location);
    const encodedModel = encodeURIComponent(model);
    return `https://${host}/v1/projects/${encodedProject}/locations/${encodedLocation}/publishers/anthropic/models/${encodedModel}:streamRawPredict`;
}

export default async function registerVertexAnthropic(pi) {
    const packageJson = findPiPackageJson();
    const requireFromPi = createRequire(packageJson);
    const { GoogleAuth } = requireFromPi("google-auth-library");
    const piAiPath = join(
        dirname(packageJson),
        "node_modules",
        "@earendil-works",
        "pi-ai",
        "dist",
        "api",
        "anthropic-messages.js",
    );
    const { streamSimple } = await import(pathToFileURL(piAiPath).href);
    const googleAuth = new GoogleAuth({ scopes: [ADC_SCOPE] });

    const vertexFetch = async (input, init = {}) => {
        const request = input instanceof Request ? input : undefined;
        const requestHeaders = new Headers(request?.headers);
        new Headers(init.headers).forEach((value, key) => requestHeaders.set(key, value));
        const rawBody = init.body ?? (request ? await request.clone().text() : undefined);
        if (typeof rawBody !== "string") {
            throw new Error("Vertex Anthropic request did not contain a JSON body");
        }
        const body = JSON.parse(rawBody);
        const model = body.model;
        if (typeof model !== "string" || !model) {
            throw new Error("Vertex Anthropic request did not specify a model");
        }
        const project = getProject();
        const location = getLocation();
        if (!project || !location) {
            throw new Error(
                "Vertex Anthropic requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION",
            );
        }

        delete body.model;
        body.anthropic_version = VERTEX_ANTHROPIC_VERSION;
        const accessToken = await googleAuth.getAccessToken();
        if (!accessToken) {
            throw new Error("Could not obtain Google Application Default Credentials for Vertex AI");
        }
        requestHeaders.delete("x-api-key");
        requestHeaders.delete("anthropic-version");
        requestHeaders.delete("content-length");
        requestHeaders.set("authorization", `Bearer ${accessToken}`);
        requestHeaders.set("content-type", "application/json");

        return fetch(getVertexUrl(project, location, model), {
            ...init,
            method: "POST",
            headers: requestHeaders,
            body: JSON.stringify(body),
            signal: init.signal ?? request?.signal,
        });
    };

    pi.registerProvider(PROVIDER_ID, {
        name: "Google Vertex AI (Anthropic)",
        api: "anthropic-messages",
        apiKey: "arkai-managed-vertex-adc",
        streamSimple(model, context, options = {}) {
            return streamSimple(model, context, {
                ...options,
                apiKey: "arkai-managed-vertex-adc",
                fetch: vertexFetch,
            });
        },
    });
}
