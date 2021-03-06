﻿using Hubbup.MikLabelModel;
using Octokit;

namespace Hubbup.Web.ViewModels
{
    public class LabelSuggestionPartialModel
    {
        public string RepoOwner { get; set; }
        public string RepoName { get; set; }
        public Issue Issue{ get; set; }
        public Label Label { get; set; }
        public LabelAreaScore Score { get; set; }
        public int Index { get; set; }
        public bool IsBestPrediction { get; set; }
    }
}
