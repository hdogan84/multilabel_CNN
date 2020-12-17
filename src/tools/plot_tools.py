import librosa
import matplotlib.pyplot as plt


def print_mel_spec(mel_spec, sample_rate, fft_hop_size_in_samples):
    # Make a new figure
    plt.figure(figsize=(12, 4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(
        mel_spec,
        sr=sample_rate,
        hop_length=fft_hop_size_in_samples,
        x_axis="time",
        y_axis="mel",
    )

    # Put a descriptive title on the plot
    plt.title("mel power spectrogram")

    # draw a color bar
    plt.colorbar(format="%+02.0f dB")

    # Make the figure layout compact
    plt.tight_layout()
