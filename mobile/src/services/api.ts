import { API_BASE_URL, REQUEST_TIMEOUT_MS } from "@/config";
import type { CapturedPhoto, CountMode, DetectionResponse } from "@/types";

export class ApiError extends Error {}

/**
 * Uploads the captured photo to the /detect endpoint and returns the parsed
 * detection result. Network failures and non-200 responses are normalized
 * into ApiError so the UI has one error shape to handle.
 */
export async function analyzeImage(
  photo: CapturedPhoto,
  mode: CountMode
): Promise<DetectionResponse> {
  const formData = new FormData();
  // React Native's fetch/FormData accepts this {uri, name, type} shape for
  // file uploads -- it isn't a real Blob, but the RN networking layer knows
  // how to stream it from the file:// uri.
  formData.append("image", {
    uri: photo.uri,
    name: "capture.jpg",
    type: "image/jpeg",
  } as unknown as Blob);
  formData.append("mode", mode);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/detect`, {
      method: "POST",
      body: formData,
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
  } catch (err) {
    if (controller.signal.aborted) {
      throw new ApiError("The request timed out. Check your connection and try again.");
    }
    throw new ApiError(
      "Could not reach the analysis server. Check that the backend is running and reachable."
    );
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    throw new ApiError(`Server error (${response.status}). Please try again.`);
  }

  return (await response.json()) as DetectionResponse;
}
