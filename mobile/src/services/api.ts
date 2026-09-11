import { File, UploadType } from "expo-file-system";

import { API_BASE_URL, REQUEST_TIMEOUT_MS } from "@/config";
import type { CapturedPhoto, CountMode, DetectionResponse } from "@/types";

export class ApiError extends Error {}

/**
 * Uploads the captured photo to the /detect endpoint and returns the parsed
 * detection result.
 *
 * Uses expo-file-system's native multipart upload (File.upload) rather than
 * fetch()+FormData: React Native's own FormData implementation only accepts
 * real Blob/File instances or strings as parts, and rejects the classic
 * `{uri, name, type}` object trick with "Unsupported FormDataPart
 * implementation" (that trick worked on older RN versions but no longer
 * does). expo-file-system's upload API is built specifically for streaming
 * a local file into a multipart request and sidesteps that entirely.
 */
export async function analyzeImage(
  photo: CapturedPhoto,
  mode: CountMode
): Promise<DetectionResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const file = new File(photo.uri);
    const result = await file.upload(`${API_BASE_URL}/detect`, {
      uploadType: UploadType.MULTIPART,
      fieldName: "image",
      mimeType: "image/jpeg",
      parameters: { mode },
      signal: controller.signal,
    });

    if (result.status < 200 || result.status >= 300) {
      throw new ApiError(`Server error (${result.status}). Please try again.`);
    }

    return JSON.parse(result.body) as DetectionResponse;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (controller.signal.aborted) {
      throw new ApiError("The request timed out. Check your connection and try again.");
    }
    throw new ApiError(
      "Could not reach the analysis server. Check that the backend is running and reachable."
    );
  } finally {
    clearTimeout(timeout);
  }
}
