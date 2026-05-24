#include <SDL2/SDL.h>
#include <SDL2/SDL_audio.h>
#include <SDL2/SDL_error.h>
#include <SDL2/SDL_timer.h>
#include <math.h>
#include <stdio.h>

#define PI 3.14159265358979323846

int fan_speed() {
  char *filename = "/proc/acpi/ibm/fan";
  FILE *fi = fopen(filename, "r");
  if (fi == NULL) {
    printf("Couldn't open file\n");
    return 1;
  }

  char buf[256];

  int rpm;
  while (fgets(buf, sizeof(buf), fi) != NULL) {
    if (sscanf(buf, "speed: %d", &rpm) == 1) {
      break;
    }
  };

  fclose(fi);
  return rpm;
}

int main() {
  SDL_InitSubSystem(SDL_INIT_AUDIO);
  SDL_AudioSpec desired = {.freq = 48000,
                           .channels = 1,
                           .format = AUDIO_F32,
                           .samples = 1024},
                obtained = {};

  SDL_AudioDeviceID device_id = SDL_OpenAudioDevice(
      NULL, 0, &desired, &obtained, SDL_AUDIO_ALLOW_ANY_CHANGE);

  if (obtained.format == 0) {
    printf("Failed to retrieve audio device: %s\n", SDL_GetError());
  } else {
    printf("Obtained audio device:\n  id: %d\n  freq: %d\n  ch: %d\n  samples: "
           "%d\n",
           device_id, obtained.freq, obtained.channels, obtained.samples);
  }

  SDL_AudioStream *stream =
      SDL_NewAudioStream(desired.format, desired.channels, desired.freq,
                         obtained.format, obtained.channels, obtained.freq);
  if (stream == NULL) {
    printf("Failed to create audio stream: %s\n", SDL_GetError());
  } else {
    printf("Stream loaded successfully at %p\n", stream);
  }

  int rpm;
  float src_samples[desired.samples]; // dst_samples[obtained.samples];

  rpm = fan_speed();
  printf("fan speed: %d\n", rpm);

  printf("Screaming...\n");
  int available = 0;
  float volume = 0.25, a_hz = 440, increment = a_hz / desired.freq * 2 * PI;
  float phase = 0;
  while (1) {
    // fill src buffer with audio data
    for (int i = 0; i < desired.samples; i++) {
      src_samples[i] = sinf(phase) * volume;
      phase += increment;
      if (phase > (2 * PI))
        phase -= 2 * PI;
    };
    SDL_AudioStreamPut(stream, src_samples, desired.samples * sizeof(float));

    // pause to allow stream to clear
    if ((available = SDL_AudioStreamAvailable(stream)) < desired.samples)
      SDL_Delay(1000);

    // fill dest stream with buffer data
  }

  SDL_FreeAudioStream(stream);
  return 0;
}
