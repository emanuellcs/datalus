import React from "react";
import { createRoot } from "react-dom/client";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import * as ort from "onnxruntime-web";

type ComponentArgs = {
  artifactBaseUrl: string;
  schema: Record<string, unknown>;
  encoder: EncoderConfig;
  projector: ProjectorConfig;
  manifest: Record<string, unknown>;
  nRecords: number;
  ddimSteps: number;
  seed: number;
  guidanceScale: number;
  conditions: Record<string, unknown>;
};

type NumericTransform = {
  quantiles: number[];
  references: number[];
};

type CategoryVocab = {
  categories: string[];
  null_token?: string;
  unknown_token?: string;
};

type EncoderConfig = {
  numeric_transforms?: Record<string, NumericTransform>;
  categorical_vocabs?: Record<string, CategoryVocab>;
};

type ProjectorConfig = {
  numerical_columns?: string[];
  categorical_columns?: string[];
  latent_dim?: number;
  cat_dims?: Array<{ cardinality: number; embedding_dim: number }>;
  embeddings?: number[][][];
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
  const latentDim = Number(
    args.projector?.latent_dim ?? (args.manifest as any)?.latent_dim ?? 16,
  );
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
  return decodeLatents(x, latentDim, args);
}

function interpolate(value: number, xs: number[], ys: number[]): number {
  if (!xs.length || !ys.length) return value;
  if (value <= xs[0]) return ys[0];
  if (value >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i += 1) {
    if (value <= xs[i]) {
      const leftX = xs[i - 1];
      const rightX = xs[i];
      const weight = (value - leftX) / Math.max(rightX - leftX, 1e-12);
      return ys[i - 1] + weight * (ys[i] - ys[i - 1]);
    }
  }
  return ys[ys.length - 1];
}

function nearestCategory(
  chunk: Float32Array,
  weights: number[][],
  vocab?: CategoryVocab,
): string | null {
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  weights.forEach((weight, idx) => {
    let distance = 0;
    for (let i = 0; i < chunk.length; i += 1) {
      const delta = chunk[i] - Number(weight[i] ?? 0);
      distance += delta * delta;
    }
    if (distance < bestDistance) {
      best = idx;
      bestDistance = distance;
    }
  });
  const unknown = vocab?.unknown_token ?? "__UNKNOWN__";
  const nullToken = vocab?.null_token ?? "__NULL__";
  const tokens = [unknown, nullToken, ...(vocab?.categories ?? [])];
  const token = tokens[best] ?? unknown;
  return token === nullToken ? null : token;
}

function decodeLatents(
  latents: Float32Array,
  latentDim: number,
  args: ComponentArgs,
): Record<string, unknown>[] {
  const numericColumns = args.projector?.numerical_columns ?? [];
  const categoricalColumns = args.projector?.categorical_columns ?? [];
  const records: Record<string, unknown>[] = [];
  for (let row = 0; row < args.nRecords; row += 1) {
    const record: Record<string, unknown> = {};
    let cursor = 0;
    numericColumns.forEach((column) => {
      const transform = args.encoder?.numeric_transforms?.[column];
      const encoded = latents[row * latentDim + cursor];
      cursor += 1;
      const unit = Math.min(1, Math.max(0, (encoded + 1) / 2));
      record[column] = transform
        ? interpolate(unit, transform.references, transform.quantiles)
        : Number(encoded.toFixed(6));
    });
    categoricalColumns.forEach((column, idx) => {
      const dim = Number(args.projector?.cat_dims?.[idx]?.embedding_dim ?? 1);
      const start = row * latentDim + cursor;
      const chunk = latents.slice(start, start + dim);
      cursor += dim;
      record[column] = nearestCategory(
        chunk,
        args.projector?.embeddings?.[idx] ?? [],
        args.encoder?.categorical_vocabs?.[column],
      );
    });
    records.push(record);
  }
  return records;
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
