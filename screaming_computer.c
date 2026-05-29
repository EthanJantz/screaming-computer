#define _POSIX_C_SOURCE 200809L
#include <SDL2/SDL.h>
#include <SDL2/SDL_audio.h>
#include <SDL2/SDL_error.h>
#include <SDL2/SDL_timer.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>

#define PI 3.14159265358979323846

float load_avg_1m(void) {
  FILE *fp;
  char buffer[256];
  char uptime_string[256] = {0};

  // Open the command for reading
  fp = popen("uptime", "r"); // depends on POSIX_C_SOURCE 200809L
  if (fp == NULL) {
    perror("Failed to run uptime");
    return 1;
  }

  float load_avg_1m = 0.0;

  // Read the output into the buffer
  if (fgets(buffer, sizeof(buffer), fp) != NULL) {
    // Copy the result into your string
    snprintf(uptime_string, sizeof(uptime_string), "%s", buffer);
    char *p = strstr(uptime_string, "load average:");
    if (p != NULL) {
      sscanf(p, "load average: %f", &load_avg_1m);
    }
  }

  // Close the pipe
  pclose(fp);
  return load_avg_1m;
}

float nproc(void) {
  FILE *fp;
  char buffer[256];

  fp = popen("nproc", "r"); // depends on POSIX_C_SOURCE 200809L
  if (fp == NULL) {
    perror("Failed to run nproc");
    return 1;
  }

  int procs = 0;

  if (fgets(buffer, sizeof(buffer), fp) != NULL) {
    sscanf(buffer, "%d", &procs);
  }

  pclose(fp);
  return procs;
}

float effort_level(void) {
  float effort = load_avg_1m() / nproc();
  return effort;
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

  const unsigned MOD_ITER = 10;
  unsigned iterations = 0;
  float cur_effort;

  cur_effort = effort_level();

  float src_samples[desired.samples], dst_samples[obtained.samples];
  int available = 0;
  float volume = 0.25, a_hz = 440, increment = a_hz / desired.freq * 2 * PI;
  float phase = 0;

  SDL_PauseAudioDevice(device_id, 0);

  signal(SIGINT, handle_sigint);
  while (running) {
    iterations++;
    if (iterations % MOD_ITER == 0) { // Prevent unnecssarily I/O
      cur_effort = effort_level();
      volume = (float)cur_effort;
      printf("effort: %f\nvolume: %f\n iter: %d\n ", cur_effort, volume,
             iterations);
    }

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
  }

  SDL_FreeAudioStream(stream);
  SDL_CloseAudioDevice(device_id);
  SDL_Quit();
  return 0;
}
