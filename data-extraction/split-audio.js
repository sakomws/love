const ffmpeg = require("fluent-ffmpeg");
const fs = require("fs");
const path = require("path");

const [inputFile] = process.argv.slice(2);

if (!inputFile) {
	console.error(`Usage: node split-audio <inputFile>

    It splits the file input.m4a into multiple files of 24 MB each.
    Try changing the MAX_FILE_SIZE and CHUNK_DURATION constants to adjust the chunk size.
    Example: node split-audio input.m4a`);
	process.exit(1);
}

// Set the input file and output directory
const MAX_FILE_SIZE = 24; // Maximum file size in MB
const CHUNK_DURATION = 40 * 60; // Starting Duration of each chunk in seconds

// Function to get the file size in MB
const getFileSizeInMB = (filePath) => {
	const stats = fs.statSync(filePath);
	return stats.size / (1024 * 1024);
};

if (getFileSizeInMB(inputFile) <= MAX_FILE_SIZE) {
	console.log("File is already smaller than the specified size");
	process.exit(0);
}

// Split the file
const splitFile = async () => {
	try {
		const tempFile = "temp_part.m4a";
		let startTime = 0;
		let partNumber = 1;
		let duration = CHUNK_DURATION; // Start with an arbitrary chunk duration (in seconds)
		const extension = path.extname(inputFile);
		const dirname = path.dirname(inputFile);
		const fileName = path.basename(inputFile, extension);

		while (true) {
			const outputPath = path.join(dirname, `${fileName}@${partNumber}.m4a`);

			await new Promise((resolve, reject) => {
				ffmpeg(inputFile)
					.setStartTime(startTime)
					.duration(duration)
					.output(tempFile)
					.on("end", () => {
						if (getFileSizeInMB(tempFile) <= MAX_FILE_SIZE) {
							fs.renameSync(tempFile, outputPath);
							startTime += duration;
							partNumber++;
							resolve();
						} else {
							duration -= 1;
							resolve(); // Adjust duration and try again
						}
					})
					.on("error", (err) => {
						reject(err);
					})
					.run();
			});

			if (startTime >= (await getAudioDuration(inputFile))) {
				break; // End loop when the entire file is processed
			}
		}

		console.log("Splitting completed!");
	} catch (error) {
		console.error("Error:", error);
	}
};

// Function to get the audio duration
const getAudioDuration = (input) => {
	return new Promise((resolve, reject) => {
		ffmpeg.ffprobe(input, (err, metadata) => {
			if (err) {
				reject(err);
			} else {
				resolve(metadata.format.duration);
			}
		});
	});
};

splitFile();
