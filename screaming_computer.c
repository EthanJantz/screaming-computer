#include <SDL2/SDL.h>
#include <SDL2/SDL_audio.h>
#include <SDL2/SDL_error.h>
#include <SDL2/SDL_timer.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>

#define PI 3.14159265358979323846

int fan_speed(void) {
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

volatile int running = 1;
void handle_sigint(int sig) { running = 0; }

int main(void) {
  SDL_InitSubSystem(SDL_INIT_AUDIO);
  SDL_AudioSpec desired = {.freq = 48000,
                           .channels = 1,
                           .format = AUDIO_F32,
                           .samples = 1024},
                obtained = {};

  SDL_AudioDeviceID device_id = SDL_OpenAudioDevice(
      NULL, 0, &desired, &obtained, SDL_AUDIO_ALLOW_ANY_CHANGE);

  if (device_id == 0) {
    printf("Failed to retrieve audio device: %s\n", SDL_GetError());
    return 1;
  } else {
    printf("Obtained audio device:\n  id: %d\n  freq: %d\n  ch: %d\n  samples: "
           "%d\n  format: %d\n",
           device_id, obtained.freq, obtained.channels, obtained.samples,
           obtained.format);
  }

  SDL_AudioStream *stream =
      SDL_NewAudioStream(desired.format, desired.channels, desired.freq,
                         obtained.format, obtained.channels, obtained.freq);
  if (stream == NULL) {
    printf("Failed to create audio stream: %s\n", SDL_GetError());
    return 1;
  } else {
    printf("Stream loaded successfully at %p\n", stream);
  }

  int rpm;

  rpm = fan_speed();
  printf("fan speed: %d\n", rpm);

  printf("Screaming...\n");
  float src_samples[desired.samples], dst_samples[obtained.samples];
  int available = 0;
  float volume = 0.25, a_hz = 440, increment = a_hz / desired.freq * 2 * PI;
  float phase = 0;

  SDL_PauseAudioDevice(device_id, 0);

  signal(SIGINT, handle_sigint);
  while (running) {
    rpm = fan_speed();
    if (rpm > 1000) {
      // pause to allow stream to clear
      available = SDL_AudioStreamAvailable(stream);
      if (available >= desired.samples)
        SDL_Delay(100);

      // create the data to send to stream
      for (int i = 0; i < desired.samples; i++) {
        src_samples[i] = sinf(phase) * volume;
        phase += increment;
        if (phase > (2 * PI))
          phase -= 2 * PI;
      };

      // send it to the stream
      int rc = SDL_AudioStreamPut(stream, src_samples,
                                  desired.samples * sizeof(float));
      if (rc == -1) {
        printf("Failed to put data into stream: %s\n", SDL_GetError());
        return 1;
      }

      // get the converted data back from the stream
      int bytes_read = SDL_AudioStreamGet(stream, dst_samples,
                                          obtained.samples * sizeof(float));
      if (bytes_read == -1) {
        printf("Failed to get data from audio stream: %s\n", SDL_GetError());
        return 1;
      }

      rc = SDL_QueueAudio(device_id, dst_samples, bytes_read);
      if (rc < 0) {
        printf("Failed to queue audio: %s\n", SDL_GetError());
        return 1;
      }
    } else {
      SDL_Delay(100);
    }
  }

  SDL_FreeAudioStream(stream);
  SDL_CloseAudioDevice(device_id);
  SDL_Quit();
  return 0;
}
