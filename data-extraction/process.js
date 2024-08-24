const fs = require("fs");
const path = require("path");

// Paths
const transcriptionsFolder = "./transcriptions";
const profilesFile = "./profiles.txt";
const outputFolder = "./output";

// Read profiles.txt and create a map of username -> linkedIn profile
const profiles = fs
	.readFileSync(profilesFile, "utf-8")
	.split("\n")
	.filter(Boolean)
	.reduce((acc, line) => {
		const [username, linkedIn] = line.split(" ");
		acc[username] = linkedIn;
		return acc;
	}, {});

// Function to process each transcription file
const processTranscriptionFile = (filename) => {
	const username = filename.split(".mp4.json")[0];
	const filePath = path.join(transcriptionsFolder, filename);
	const transcriptionData = JSON.parse(fs.readFileSync(filePath, "utf-8"));

	const outputData = {
		user: username,
		linkedIn: profiles[username] || "Profile not found",
		transcription: transcriptionData.text,
	};

	const outputFilePath = path.join(outputFolder, `${username}.json`);
	fs.writeFileSync(outputFilePath, JSON.stringify(outputData, null, 2));
};

// Main function
const main = () => {
	// Ensure the output directory exists
	if (!fs.existsSync(outputFolder)) {
		fs.mkdirSync(outputFolder);
	}

	// Read all transcription files
	const transcriptionFiles = fs
		.readdirSync(transcriptionsFolder)
		.filter((file) => file.endsWith(".json"));

	// Process each transcription file
	transcriptionFiles.forEach(processTranscriptionFile);
};

main();
