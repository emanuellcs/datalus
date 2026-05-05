import React from "react";
import { createRoot } from "react-dom/client";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import * as ort from "onnxruntime-web";

type ComponentArgs = {
  artifactBaseUrl: string;
  schema: Record<string, unknown>;
  nRecords: number;
  ddimSteps: number;
  seed: number;
  guidanceScale: number;
  conditions: Record<string, unknown>;
};

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function gaussian(random: () => number): number {
  const u1 = Math.max(random(), 1e-7);
  const u2 = random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function cosineSchedule(numTimesteps: number): Float32Array {
  const values = new Float32Array(numTimesteps);
  const s = 0.008;
  let first = 1.0;
  for (let i = 0; i < numTimesteps; i += 1) {
    const x = i / numTimesteps;
    const alpha = Math.cos(((x + s) / (1 + s)) * Math.PI * 0.5) ** 2;
    if (i === 0) first = alpha;
    values[i] = alpha / first;
  }
  return values;
}

async function loadArrayBuffer(url: string): Promise<ArrayBuffer> {
  const cache = await caches.open("datalus-onnx-artifacts");
  const cached = await cache.match(url);
  if (cached) return cached.arrayBuffer();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Falha ao baixar artefato: ${url}`);
  await cache.put(url, response.clone());
  return response.arrayBuffer();
}

async function runDdim(
  args: ComponentArgs,
): Promise<Record<string, unknown>[]> {
  const precision = String(args.conditions?.precision ?? "model_int8.onnx");
  const modelBytes = await loadArrayBuffer(
    `${args.artifactBaseUrl}/${precision}`,
  );
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const input = session.inputNames[0];
  const timestep = session.inputNames[1];
  const latentDim = Number((args.schema as any)?.latent_dim ?? 16);
  const random = seededRandom(args.seed);
  const total = args.nRecords * latentDim;
  let x = new Float32Array(total);
  for (let i = 0; i < total; i += 1) x[i] = gaussian(random);
  const alphas = cosineSchedule(1000);
  const stepRatio = Math.floor(1000 / args.ddimSteps);
  for (let i = args.ddimSteps - 1; i >= 0; i -= 1) {
    const t = Math.min(999, i * stepRatio);
    const prev = i > 0 ? Math.min(999, (i - 1) * stepRatio) : -1;
    const feeds: Record<string, ort.Tensor> = {
      [input]: new ort.Tensor("float32", x, [args.nRecords, latentDim]),
      [timestep]: new ort.Tensor(
        "int64",
        BigInt64Array.from({ length: args.nRecords }, () => BigInt(t)),
        [args.nRecords],
      ),
    };
    const output = await session.run(feeds);
    const noise = output[session.outputNames[0]].data as Float32Array;
    const alphaT = Math.max(alphas[t], 1e-8);
    const alphaPrev = prev >= 0 ? Math.max(alphas[prev], 1e-8) : 1.0;
    const next = new Float32Array(total);
    for (let j = 0; j < total; j += 1) {
      const predX0 =
        (x[j] - Math.sqrt(1 - alphaT) * noise[j]) / Math.sqrt(alphaT);
      next[j] =
        Math.sqrt(alphaPrev) * predX0 +
        Math.sqrt(Math.max(1 - alphaPrev, 0)) * noise[j];
    }
    x = next;
  }
  const columns = Object.keys(args.schema || {});
  return Array.from({ length: args.nRecords }, (_, row) => {
    const record: Record<string, unknown> = {};
    columns.forEach((column, idx) => {
      record[column] = Number(
        x[row * latentDim + (idx % latentDim)].toFixed(6),
      );
    });
    return record;
  });
}

function Component(props: { args: ComponentArgs }): JSX.Element {
  const [status, setStatus] = React.useState("Pronto");
  React.useEffect(() => {
    Streamlit.setFrameHeight(96);
  }, []);
  const onClick = async () => {
    try {
      setStatus("Executando no navegador...");
      const records = await runDdim(props.args);
      Streamlit.setComponentValue({ status: "completed", records });
      setStatus("Concluído");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      Streamlit.setComponentValue({ status: "failed", error: message });
      setStatus(message);
    }
  };
  return (
    <div style={{ fontFamily: "system-ui, sans-serif", padding: "0.5rem 0" }}>
      <button onClick={onClick}>Executar inferência local</button>
      <span style={{ marginLeft: "0.75rem" }}>{status}</span>
    </div>
  );
}

const Connected = withStreamlitConnection(Component);
createRoot(document.getElementById("root") as HTMLElement).render(
  <Connected />,
);
