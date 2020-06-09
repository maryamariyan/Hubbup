﻿using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CreateMikLabelModel.ML
{
    public static class DatasetHelper
    {
        public static async Task PrepareAndSaveDatasetsForIssuesAsync(DataFilePaths issueFiles, DatasetModifier datasetModifier)
        {
            var ds = DatasetHelperInner.Instance;

            var lines = await ds.AddOrRemoveColumnsPriorToTrainingAsync(issueFiles.InputPath, datasetModifier, includeFileColumns: false);

            lines = ds.OnlyIssues(lines);
            await ds.BreakIntoTrainValidateTestDatasetsAsync(lines, issueFiles.TrainPath, issueFiles.ValidatePath, issueFiles.TestPath);
        }

        public static async Task PrepareAndSaveDatasetsForPrsAsync(DataFilePaths prFiles, DatasetModifier datasetModifier)
        {
            var ds = DatasetHelperInner.Instance;

            var lines = await ds.AddOrRemoveColumnsPriorToTrainingAsync(prFiles.InputPath, datasetModifier, includeFileColumns: true);

            lines = ds.OnlyPrs(lines);
            await ds.BreakIntoTrainValidateTestDatasetsAsync(lines, prFiles.TrainPath, prFiles.ValidatePath, prFiles.TestPath);
        }

        private class DatasetHelperInner
        {
            public static readonly DatasetHelperInner Instance = new DatasetHelperInner(new DiffHelper());
            private DatasetHelperInner(DiffHelper diffHelper)
            {
                _diffHelper = diffHelper;
                _sb = new StringBuilder();
                _folderSb = new StringBuilder();
                _regexForUserMentions = new Regex(@"@[a-zA-Z0-9_//-]+");
            }
            private readonly Regex _regexForUserMentions;
            private readonly StringBuilder _folderSb;
            private readonly DiffHelper _diffHelper;
            private readonly StringBuilder _sb;

            /// <summary>
            /// partitions the dataset in inputPath into train, validate and test datapaths
            /// </summary>
            /// <param name="inputPath">path to the input dataset</param>
            /// <param name="trainPath">the output to store the train dataset</param>
            /// <param name="validatePath">the output to store the train dataset</param>
            /// <param name="testPath">the output to store the train dataset</param>
            public async Task BreakIntoTrainValidateTestDatasetsAsync(string[] lines, string trainPath, string validatePath, string testPath)
            {
                int totalCount = lines.Length;

                // have at least 1000 elements
                Debug.Assert(totalCount > 1000);
                int numInTrain = (int)(lines.Length * 0.8);
                int numInValidate = (int)(lines.Length * 0.1);

                // 80% into train dataset
                await SaveFromXToYAsync(
                    lines,
                    trainPath,
                    numToSkip: 0, length: numInTrain);

                // next 10% into validate dataset
                await SaveFromXToYAsync(
                    lines,
                    validatePath,
                    numToSkip: numInTrain, length: numInValidate); // next 10%

                // remaining 10% into test dataset
                await SaveFromXToYAsync(
                    lines,
                    testPath,
                    numToSkip: numInTrain + numInValidate);
            }

            private async Task SaveFromXToYAsync(string[] lines, string output, int numToSkip, int length = -1)
            {
                var header = lines.Take(1).ToArray(); // include header
                await File.WriteAllLinesAsync(output, header);
                lines = lines.Skip(numToSkip + 1).ToArray();
                if (length != -1)
                {
                    lines = lines.Take(length).ToArray(); // include header
                }
                await File.AppendAllLinesAsync(output, lines);
            }

            /// <summary>
            /// saves to file a subset containing only PRs
            /// </summary>
            /// <param name="input">path to the reference dataset</param>
            /// <param name="output">the output to store the new dataset</param>
            public string[] OnlyPrs(string[] lines)
            {
                var header = lines.Take(1).ToArray(); // include header
                Debug.Assert(header[0].Split("\t")[_isPrIndex] == "IsPR");
                lines = lines.Skip(1).ToArray();
                return header.Union(
                    lines.Where(x => int.TryParse(x.Split('\t')[_isPrIndex], out int isPrAsNumber) && isPrAsNumber == 1)).ToArray();
            }

            private readonly int _isPrIndex = 6;

            /// <summary>
            /// saves to file a subset containing only issues
            /// </summary>
            /// <param name="input">path to the reference dataset</param>
            /// <param name="output">the output to store the new dataset</param>
            public string[] OnlyIssues(string[] lines)
            {
                var header = lines.Take(1).ToArray(); // include header
                Debug.Assert(header[0].Split("\t")[_isPrIndex] == "IsPR");
                lines = lines.Skip(1).ToArray();
                return header.Union(
                    lines.Where(x => int.TryParse(x.Split('\t')[_isPrIndex], out int isPrAsNumber) && isPrAsNumber == 0)).ToArray();
            }

            /// <summary>
            /// saves to file a dataset ready for training, given one created using GithubIssueDownloader.
            /// For training we can remove ID column, and further expand information in FilePaths
            /// We also retrieve user @ mentions from instead Description and add into new columns
            /// </summary>
            /// <param name="input">path to the reference dataset</param>
            /// <param name="output">the output to store the new dataset</param>
            /// <param name="includeFileColumns">when true, it contains extra columns with file related information</param>
            /// <param name="reMapFiles">for PRs in archived repos, how they could be re-mapped if they were transferred</param>
            public async Task<string[]> AddOrRemoveColumnsPriorToTrainingAsync(
                string input,
                DatasetModifier datasetModifier,
                bool includeFileColumns = true)
            {
                var existingHeaders = 
                    new string[] { "CombinedID", "ID", "Area", "Title", "Description", "Author", "IsPR", "FilePaths" };
                var headersToKeep = 
                    new string[] { "CombinedID", "ID", "Area", "Title", "Description", "Author", "IsPR" };
                var newOnesToAdd = 
                    new string[] { "NumMentions", "UserMentions" };

                var headerIndices = new Dictionary<string, int>();
                for (int i = 0; i < existingHeaders.Length; i++)
                {
                    headerIndices.Add(existingHeaders[i], i);
                }

                var sbInner = new StringBuilder();
                foreach (var item in headersToKeep.Union(newOnesToAdd.SkipLast(1)))
                {
                    sbInner.Append(item).Append("\t");
                }
                sbInner.Append(newOnesToAdd.Last());
                if (includeFileColumns)
                {
                    if (datasetModifier.ReMapLabel == null)
                    {
                        throw new InvalidOperationException(nameof(datasetModifier));
                    }
                    sbInner.Append("\tFileCount\tFiles\tFilenames\tFileExtensions\tFolderNames\tFolders");
                }
                var newHeader = sbInner.ToString();

                var newLines = new List<string>();
                newLines.Add(newHeader);

                var lines = await File.ReadAllLinesAsync(input);
                string body;
                if (lines.Length != 0)
                {
                    foreach (var line in lines.Where(x => !x.StartsWith("CombinedID") && !string.IsNullOrEmpty(x)))
                    {
                        _sb.Clear();
                        var lineSplitByTab = line.Split("\t");
                        string fromRepo = lineSplitByTab[headerIndices["CombinedID"]].Split(",")[1];
                        string area = datasetModifier.ReMapLabel(lineSplitByTab[headerIndices["Area"]], fromRepo);
                        if (string.IsNullOrWhiteSpace(area))
                        {
                            // the label from archived file is not being used in targetRepo.. can skip this row
                            continue;
                        }

                        _sb
                            .Append(lineSplitByTab[headerIndices["CombinedID"]])
                            .Append('\t').Append(lineSplitByTab[headerIndices["ID"]])
                            .Append('\t').Append(area)
                            .Append('\t').Append(lineSplitByTab[headerIndices["Title"]]);

                        body = lineSplitByTab[headerIndices["Description"]];
                        _sb.Append('\t').Append(body);
                        _sb.Append('\t').Append(lineSplitByTab[headerIndices["Author"]]);

                        int.TryParse(lineSplitByTab[headerIndices["IsPR"]], out int isPrAsNumber);
                        Debug.Assert((isPrAsNumber == 1 || isPrAsNumber == 0));
                        _sb.Append('\t').Append(isPrAsNumber);

                        AppendColumnsForUserMentions(body);
                        if (includeFileColumns)
                        {
                            AppendColumnsForFileDiffs(lineSplitByTab[headerIndices["FilePaths"]], isPr: isPrAsNumber == 1, datasetModifier.ReMapFiles, fromRepo);
                        }
                        newLines.Add(_sb.ToString());
                    }
                }

                return newLines.ToArray();
            }

            private void AppendColumnsForUserMentions(string body)
            {
                var userMentions = _regexForUserMentions.Matches(body).Select(x => x.Value).ToArray();
                _sb.Append('\t').Append(userMentions.Length)
                    .Append('\t').Append(FlattenIntoColumn(userMentions));
            }

            private void AppendColumnsForFileDiffs(string semicolonDelimitedFilesWithDiff, bool isPr, Func<string[], string, string[]> reMapFiles, string fromRepo)
            {
                if (isPr)
                {
                    string[] filePaths = semicolonDelimitedFilesWithDiff.Split(';');
                    int numFilesChanged = filePaths.Length == 1 && string.IsNullOrEmpty(filePaths[0]) ? 0 : filePaths.Length;
                    _sb.Append('\t').Append(numFilesChanged);
                    if (numFilesChanged != 0)
                    {
                        // for PRs in archived repos, how they could be re-mapped if they were transferred
                        filePaths = reMapFiles(filePaths, fromRepo);

                        var segmentedDiff = _diffHelper.SegmentDiff(filePaths);

                        _sb.Append('\t').Append(FlattenIntoColumn(filePaths))
                            .Append('\t').Append(FlattenIntoColumn(segmentedDiff.Filenames))
                            .Append('\t').Append(FlattenIntoColumn(segmentedDiff.Extensions))
                            .Append('\t').Append(FlattenIntoColumn(segmentedDiff.FolderNames))
                            .Append('\t').Append(FlattenIntoColumn(segmentedDiff.Folders));
                    }
                    else
                    {
                        _sb.Append('\t', 5);
                    }
                }
                else
                {
                    _sb.Append('\t').Append(0)
                        .Append('\t', 5);
                }
            }

            /// <summary>
            /// flattens a dictionary to be repeated in a space separated format
            /// </summary>
            /// <param name="textToFlatten">a dictionary containing text and number of times they were repeated</param>
            /// <returns>space delimited text</returns>
            public string FlattenIntoColumn(Dictionary<string, int> folder)
            {
                _folderSb.Clear();
                string res;
                foreach (var f in folder.OrderByDescending(x => x.Value))
                {
                    Debug.Assert(f.Value >= 1);
                    _folderSb.Append(f.Key);
                    for (int j = 0; j < f.Value - 1; j++)
                    {
                        _folderSb.Append(" ").Append(f.Key);
                    }
                    _folderSb.Append(" ");
                }
                if (_folderSb.Length == 0)
                {
                    res = string.Empty;
                }
                else
                {
                    res = _folderSb.ToString();
                    res = res.Substring(0, res.Length - 1);
                }
                return res;
            }

            /// <summary>
            /// flattens texts in a space separated format
            /// </summary>
            /// <param name="array">the input containing text to show</param>
            /// <returns>space delimited text</returns>
            public string FlattenIntoColumn(string[] array)
            {
                return string.Join(' ', array);
            }

            /// <summary>
            /// flattens texts in a space separated format
            /// </summary>
            /// <param name="enumerable">the input containing text to show</param>
            /// <returns>space delimited text</returns>
            public string FlattenIntoColumn(IEnumerable<string> enumerable)
            {
                return string.Join(' ', enumerable);
            }
        }
    }
}
