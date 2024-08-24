const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");

// Set the input and output directories
const inputDir = "./input-audio";
const outputDir = "./transcriptions";

async function executeCommand(command) {
	return new Promise((resolve, reject) => {
		exec(command, (error, stdout, stderr) => {
			if (error) {
				reject(`Error: ${error.message}`);
				return;
			}
			if (stderr) {
				// reject(`Stderr: ${stderr}`);
				resolve(stdout);
				return;
			}
			resolve(stdout);
		});
	});
}

/**
 * Reads all files in a folder and returns them as an array.
 * @param {string} folderPath - The path to the folder to read.
 * @returns {Promise<string[]>} - A promise that resolves to an array of file names.
 */
const getFilesInFolder = async (folderPath, extension = "") => {
	return new Promise((resolve, reject) => {
		fs.readdir(folderPath, (err, files) => {
			if (err) {
				return reject(err);
			}

			// Filter files based on the extension if provided
			const filePaths = files
				.filter((file) => !extension || path.extname(file) === extension)
				.map((file) => path.join(folderPath, file));

			resolve(filePaths);
		});
	});
};

const transcribe = async (inputFileName, outputFile) => {
	const command = `node transcribe-audio ${inputFileName} ${outputFile}`;
	console.log(`EXEC: ${command}`);
	await executeCommand(command);
};

(async () => {
	const files = await getFilesInFolder(inputDir, ".mp4");
	for (const file of files) {
		const inputFilePath = file;
		const outputFile = path.join(outputDir, `${path.basename(file)}.json`);
		//continue if file exists
		if (fs.existsSync(outputFile)) {
			console.log(`File ${outputFile} already exists, skipping...`);
			continue;
		}
		await transcribe(inputFilePath, outputFile);
	}
})();
