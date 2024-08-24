const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");

// Set the input and output directories
const inputDir = "./output-audio";

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

const splitAudio = async (inputFile) => {
	const command = `node split-audio ${inputFile}`;
	console.log(`EXEC: ${command}`);
	await executeCommand(command);

	//delete the inputFile
	fs.unlinkSync(inputFile);
};

(async () => {
	const files = await getFilesInFolder(inputDir, ".m4a");
	for (const file of files) {
		const inputFilePath = file;

		await splitAudio(inputFilePath);
	}
})();
