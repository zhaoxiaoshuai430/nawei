class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const requestedSize = Number(options.processorOptions?.chunkSize || 2048);
    this.chunkSize = Math.max(128, Math.round(requestedSize));
    this.buffer = new Float32Array(this.chunkSize);
    this.offset = 0;
  }

  process(inputs, outputs) {
    const output = outputs[0]?.[0];
    if (output) output.fill(0);

    const input = inputs[0]?.[0];
    if (!input?.length) return true;

    let inputOffset = 0;
    while (inputOffset < input.length) {
      const copyLength = Math.min(input.length - inputOffset, this.chunkSize - this.offset);
      this.buffer.set(input.subarray(inputOffset, inputOffset + copyLength), this.offset);
      this.offset += copyLength;
      inputOffset += copyLength;
      if (this.offset === this.chunkSize) {
        const chunk = this.buffer;
        this.port.postMessage(chunk.buffer, [chunk.buffer]);
        this.buffer = new Float32Array(this.chunkSize);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
